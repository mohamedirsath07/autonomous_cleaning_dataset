from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime
import re

app = FastAPI()

class ProfileRequest(BaseModel):
    file_path: str

class PipelineStep(BaseModel):
    column: str
    action: str  # impute, scale, encode, drop, normalize
    method: Optional[str] = None # mean, median, standard, minmax, onehot, label
    params: Optional[Dict[str, Any]] = None

class PrepareRequest(BaseModel):
    file_path: str
    steps: List[PipelineStep]

@app.post("/profile")
async def generate_profile(request: ProfileRequest):
    file_path = request.file_path
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found at {file_path}")
    
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Replace NaN with None for JSON serialization
        df_replace = df.replace({np.nan: None})

        # Basic profiling (Phase 1)
        profile = {
            "columns": [],
            "rows": len(df),
            "preview": df_replace.head(20).to_dict(orient='records')
        }
        
        for col in df.columns:
            col_data = df[col]
            
            # Determine type
            dtype = str(col_data.dtype)
            inferred_type = "text"
            if pd.api.types.is_numeric_dtype(col_data):
                inferred_type = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                inferred_type = "datetime"
            
            # Issue Detection
            issues = []
            suggested = []
            
            missing_pct = float(col_data.isnull().mean() * 100)
            if missing_pct > 0:
                issues.append("missing")
                if inferred_type == "numeric":
                    suggested.append("impute_median")
                else:
                    suggested.append("impute_mode")
            
            col_profile = {
                "name": col,
                "inferred_type": inferred_type,
                "dtype": dtype,
                "missing_pct": missing_pct,
                "unique": int(col_data.nunique()),
                "sample_values": col_data.dropna().head(5).tolist(),
                "issues": issues,
                "suggested": suggested
            }
            
            if inferred_type == "numeric":
                # Outlier detection using IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers_count = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
                
                col_profile.update({
                    "min": float(col_data.min()) if not col_data.empty else None,
                    "max": float(col_data.max()) if not col_data.empty else None,
                    "mean": float(col_data.mean()) if not col_data.empty else None,
                    "median": float(col_data.median()) if not col_data.median() is pd.NA else None,
                    "std": float(col_data.std()) if not col_data.empty else None,
                    "outliers_count": int(outliers_count)
                })
                
                if outliers_count > 0:
                    col_profile["issues"].append("outliers")
                    col_profile["suggested"].append("robust_scale")
                    
                # Check skewness
                if len(col_data.dropna()) > 0:
                    skewness = col_data.skew()
                    if abs(skewness) > 1:
                        col_profile["issues"].append("skewed")
                        col_profile["skewness"] = float(skewness)
            
            elif inferred_type == "text":
                # Check for categorical imbalance
                value_counts = col_data.value_counts()
                if len(value_counts) > 0 and len(value_counts) < 50:  # Likely categorical
                    max_freq = value_counts.iloc[0]
                    total = len(col_data.dropna())
                    if total > 0 and max_freq / total > 0.8:
                        col_profile["issues"].append("imbalanced")
                        
                # Suggest encoding for categorical
                if len(value_counts) < 20 and len(value_counts) > 1:
                    col_profile["suggested"].append("encode")
            
            profile["columns"].append(col_profile)
            
        return profile

    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prepare")
async def prepare_dataset(request: PrepareRequest):
    file_path = request.file_path
    steps = request.steps
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found at {file_path}")
        
    try:
        # Load Data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        transformation_log = []
        
        # Apply Steps
        for step in steps:
            col = step.column
            if col not in df.columns and step.action != 'drop': # Drop might be called on already dropped col?
                continue
                
            if step.action == "drop":
                df.drop(columns=[col], inplace=True)
                transformation_log.append(f"Dropped column: {col}")
                
            elif step.action == "impute":
                if step.method == "mean":
                    val = df[col].mean()
                    df[col] = df[col].fillna(val)
                    transformation_log.append(f"Imputed {col} with mean: {val}")
                elif step.method == "median":
                    val = df[col].median()
                    df[col] = df[col].fillna(val)
                    transformation_log.append(f"Imputed {col} with median: {val}")
                elif step.method == "mode":
                    val = df[col].mode()[0]
                    df[col] = df[col].fillna(val)
                    transformation_log.append(f"Imputed {col} with mode: {val}")
                elif step.method == "custom":
                    val = step.params.get('value')
                    df[col] = df[col].fillna(val)
                    transformation_log.append(f"Imputed {col} with custom value: {val}")

            elif step.action == "scale":
                if step.method == "standard":
                    scaler = StandardScaler()
                    df[[col]] = scaler.fit_transform(df[[col]])
                    transformation_log.append(f"StandardScaled column: {col}")
                elif step.method == "minmax":
                    scaler = MinMaxScaler()
                    df[[col]] = scaler.fit_transform(df[[col]])
                    transformation_log.append(f"MinMaxScaled column: {col}")

            elif step.action == "encode":
                if step.method == "onehot":
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(columns=[col], inplace=True)
                    transformation_log.append(f"OneHotEncoded column: {col}")
                elif step.method == "label":
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    transformation_log.append(f"LabelEncoded column: {col}")
                    
        # Save Cleaned File
        base, ext = os.path.splitext(file_path)
        cleaned_path = f"{base}_cleaned{ext}"
        
        if ext == '.csv':
            df.to_csv(cleaned_path, index=False)
        else:
            df.to_excel(cleaned_path, index=False)
        
        # Generate Python reproducible code
        python_code = generate_pipeline_code(file_path, steps)
            
        return {
            "cleaned_file_path": cleaned_path,
            "transformation_log": transformation_log,
            "preview": df.replace({np.nan: None}).head(20).to_dict(orient='records'),
            "columns": list(df.columns),
            "python_code": python_code
        }

    except Exception as e:
        print(f"Error preparing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_pipeline_code(file_path, steps):
    """Generate reproducible Python code for the pipeline"""
    code_lines = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder",
        "",
        "# Load dataset",
        f"df = pd.read_csv('{os.path.basename(file_path)}')",
        "",
        "# Apply transformations"
    ]
    
    for step in steps:
        col = step.column
        action = step.action
        method = step.method
        
        if action == "drop":
            code_lines.append(f"df.drop(columns=['{col}'], inplace=True)")
        elif action == "impute":
            if method == "mean":
                code_lines.append(f"df['{col}'].fillna(df['{col}'].mean(), inplace=True)")
            elif method == "median":
                code_lines.append(f"df['{col}'].fillna(df['{col}'].median(), inplace=True)")
            elif method == "mode":
                code_lines.append(f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)")
        elif action == "scale":
            if method == "standard":
                code_lines.append(f"scaler = StandardScaler()")
                code_lines.append(f"df[['{col}']] = scaler.fit_transform(df[['{col}']])")
            elif method == "minmax":
                code_lines.append(f"scaler = MinMaxScaler()")
                code_lines.append(f"df[['{col}']] = scaler.fit_transform(df[['{col}']])")
        elif action == "encode":
            if method == "onehot":
                code_lines.append(f"dummies = pd.get_dummies(df['{col}'], prefix='{col}')")
                code_lines.append(f"df = pd.concat([df, dummies], axis=1)")
                code_lines.append(f"df.drop(columns=['{col}'], inplace=True)")
            elif method == "label":
                code_lines.append(f"le = LabelEncoder()")
                code_lines.append(f"df['{col}'] = le.fit_transform(df['{col}'].astype(str))")
    
    code_lines.append("")
    code_lines.append("# Save cleaned dataset")
    code_lines.append("df.to_csv('cleaned_dataset.csv', index=False)")
    
    return "\n".join(code_lines)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
