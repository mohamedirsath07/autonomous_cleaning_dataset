from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from datetime import datetime
import re
import os
import requests
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_file_if_url(file_path: str) -> str:
    """Download file from URL if it's a URL, otherwise return path as-is."""
    if file_path.startswith('http://') or file_path.startswith('https://'):
        logger.info(f"Downloading file from URL: {file_path}")
        try:
            response = requests.get(file_path, timeout=30)
            response.raise_for_status()
            logger.info(f"Downloaded {len(response.content)} bytes")
            
            # Determine extension
            ext = '.csv'
            if '.xlsx' in file_path:
                ext = '.xlsx'
            elif '.xls' in file_path:
                ext = '.xls'
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            temp_file.write(response.content)
            temp_file.close()
            logger.info(f"Saved to temp file: {temp_file.name}")
            return temp_file.name
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    return file_path

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

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

class AutoCleanRequest(BaseModel):
    file_path: str
    split_ratio: float = 0.8
    k_folds: int = 5
    epochs: int = 100
    remove_outliers: bool = True
    impute_missing: bool = True
    normalize_features: bool = True


MISSING_TOKENS = {"na", "n/a", "none", "null", "?", "unknown", "nan", "", "not available"}
BOOLEAN_TRUE = {"true", "yes", "y", "1", "t"}
BOOLEAN_FALSE = {"false", "no", "n", "0", "f"}
SEMANTIC_MAPPINGS = {
    "gender": {
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
        "woman": "female",
        "man": "male",
        "other": "other",
        "non-binary": "non_binary",
    },
    "boolean": {
        **{token: "yes" for token in BOOLEAN_TRUE},
        **{token: "no" for token in BOOLEAN_FALSE},
    }
}
POSITIVE_KEYWORDS = ["age", "amount", "price", "salary", "revenue", "distance", "quantity", "count", "score", "duration"]
PERCENTAGE_KEYWORDS = ["percent", "percentage", "ratio", "rate"]
TEXT_KEYWORDS = ["description", "notes", "comment", "text", "summary", "review"]


def _snake_case(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).lower()
    cleaned = cleaned.strip("_") or "col"
    return cleaned


def _convert_numeric(series: pd.Series) -> Tuple[pd.Series, bool]:
    numeric = pd.to_numeric(series, errors="coerce")
    ratio = numeric.notna().mean() if len(series) else 0
    if ratio >= 0.8:
        return numeric, True
    return series, False


def _convert_datetime(series: pd.Series) -> Tuple[pd.Series, bool]:
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        ratio = parsed.notna().mean() if len(series) else 0
        if ratio >= 0.6:
            return parsed, True
    except Exception:
        pass
    return series, False


def _boolean_normalization(series: pd.Series) -> Tuple[pd.Series, bool]:
    true_tokens = {"true", "yes", "y", "1"}
    false_tokens = {"false", "no", "n", "0"}
    lowered = series.astype(str).str.lower().str.strip()
    if lowered.isin(true_tokens.union(false_tokens)).mean() >= 0.9:
        bool_values = lowered.isin(true_tokens).astype(int)
        return bool_values, True
    return series, False


def _log_skewness(series: pd.Series) -> bool:
    valid = series.dropna()
    if len(valid) == 0:
        return False
    skewness = valid.skew()
    return bool(abs(skewness) > 1 and (valid >= 0).all())


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip()
    as_str = as_str.str.replace(r"[\$€£¥₹,]", "", regex=True)
    as_str = as_str.str.replace(r"\s", "", regex=True)
    as_str = as_str.str.replace(r"[^0-9\.\-]", "", regex=True)
    as_str = as_str.replace({"": np.nan, "-": np.nan, ".": np.nan})
    return pd.to_numeric(as_str, errors="coerce")


def _detect_id_column(series: pd.Series) -> bool:
    if not len(series):
        return False
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    sample = series.dropna().astype(str)
    if sample.empty:
        return False
    if sample.str.match(r"^[0-9]+$").mean() < 0.9:
        return False
    unique_ratio = sample.nunique() / max(len(sample), 1)
    return bool(unique_ratio >= 0.9 and sample.str.startswith("0").any())


def _infer_column_role(name: str, series: pd.Series) -> str:
    lowered = name.lower()
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    avg_length = series.dropna().astype(str).str.len().mean() if hasattr(series, "str") else 0

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime_feature"
    if _detect_id_column(series):
        return "id"
    if any(keyword in lowered for keyword in TEXT_KEYWORDS) or (pd.api.types.is_object_dtype(series) and avg_length and avg_length > 40):
        return "text"
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        if unique_ratio > 0.85:
            return "numeric_key"
        return "numeric_feature"
    if pd.api.types.is_object_dtype(series):
        if unique_ratio > 0.5:
            return "categorical_high_cardinality"
        return "categorical_feature"
    return "unknown"


def _column_contains_keyword(name: str, keywords: List[str]) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in keywords)


def _compute_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    total_rows = len(df)
    column_metrics = {}
    for col in df.columns:
        series = df[col]
        completeness = 1 - float(series.isna().mean())
        uniqueness = float(series.nunique(dropna=True) / max(total_rows, 1))
        dtype = str(series.dtype)
        column_metrics[col] = {
            "dtype": dtype,
            "completeness": completeness,
            "uniqueness": uniqueness
        }
    return {
        "rows": total_rows,
        "columns": len(df.columns),
        "per_column": column_metrics
    }


def _auto_schema_cleaning(df: pd.DataFrame, log: List[str], actions: Dict[str, Any]) -> pd.DataFrame:
    rename_map = {}
    used_names = set()
    for col in df.columns:
        candidate = _snake_case(col)
        original = candidate
        suffix = 1
        while candidate in used_names:
            candidate = f"{original}_{suffix}"
            suffix += 1
        if candidate != col:
            rename_map[col] = candidate
        used_names.add(candidate)
    if rename_map:
        df = df.rename(columns=rename_map)
        log.append(f"Schema Cleaning: renamed columns {json.dumps(rename_map)}")
    actions["schema_cleaning"] = {"renamed": rename_map}

    type_conversions = []
    column_roles = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_object_dtype(series):
            if _detect_id_column(series):
                df[col] = series.astype(str)
                column_roles[col] = "id"
                continue

            numeric_candidate = _clean_numeric_series(series)
            if numeric_candidate.notna().mean() >= 0.8:
                df[col] = numeric_candidate
                type_conversions.append({"column": col, "target_type": "numeric"})
                column_roles[col] = "numeric_feature"
                continue

            converted, changed = _convert_datetime(series)
            if changed:
                df[col] = converted
                type_conversions.append({"column": col, "target_type": "datetime"})
                column_roles[col] = "datetime_feature"
                continue

            converted, changed = _boolean_normalization(series)
            if changed:
                df[col] = converted
                type_conversions.append({"column": col, "target_type": "boolean"})
                column_roles[col] = "numeric_feature"
                continue

            df[col] = series.astype(str).str.strip()

        column_roles[col] = column_roles.get(col) or _infer_column_role(col, df[col])

    if type_conversions:
        log.append(f"Schema Cleaning: coerced types {type_conversions}")
    actions["schema_cleaning"]["type_conversions"] = type_conversions
    actions["schema_cleaning"]["roles"] = column_roles
    return df


def _auto_semantic_cleaning(df: pd.DataFrame, log: List[str], actions: Dict[str, Any]) -> pd.DataFrame:
    roles = actions.get("schema_cleaning", {}).get("roles", {})
    semantics = {
        "dates": [],
        "categorical_standardized": [],
        "missing_tokens": [],
        "canonical_mappings": {}
    }

    for col in df.columns:
        series = df[col]
        role = roles.get(col, "")
        if pd.api.types.is_datetime64_any_dtype(series):
            df[col] = series.dt.tz_localize(None)
            semantics["dates"].append(col)
            continue

        if pd.api.types.is_object_dtype(series):
            standardized = series.astype(str).str.strip()
            lowered = standardized.str.lower()

            sentinel_mask = lowered.isin(MISSING_TOKENS)
            if sentinel_mask.any():
                standardized = standardized.mask(sentinel_mask, np.nan)
                semantics["missing_tokens"].append(col)

            if standardized.str.contains(r"\s{2,}").any():
                standardized = standardized.str.replace(r"\s+", " ", regex=True)

            applied_mapping = None
            if role == "text":
                df[col] = standardized
                continue

            if any(keyword in col.lower() for keyword in ["gender", "sex"]):
                mapping = SEMANTIC_MAPPINGS.get("gender")
                normalized = standardized.str.lower().map(mapping).fillna(standardized.str.lower())
                df[col] = normalized
                applied_mapping = "gender"
            elif lowered.isin(BOOLEAN_TRUE.union(BOOLEAN_FALSE)).mean() > 0.6:
                mapping = SEMANTIC_MAPPINGS.get("boolean")
                normalized = lowered.map(mapping).fillna(lowered)
                df[col] = normalized
                applied_mapping = "boolean"
            else:
                df[col] = standardized

            semantics["categorical_standardized"].append(col)
            if applied_mapping:
                semantics["canonical_mappings"][col] = applied_mapping

    if any(semantics.values()):
        log.append(f"Semantic Cleaning: normalized {semantics}")
    actions["semantic_cleaning"] = semantics
    return df


def _auto_missing_data(df: pd.DataFrame, log: List[str], actions: Dict[str, Any]) -> pd.DataFrame:
    roles = actions.get("schema_cleaning", {}).get("roles", {})
    imputation_records = []
    dropped_columns = []
    indicator_columns = []

    for col in list(df.columns):
        series = df[col]
        missing_ratio = series.isna().mean()
        if missing_ratio == 0:
            continue

        role = roles.get(col, "")
        if missing_ratio > 0.6 and role not in {"id", "datetime_feature", "text"}:
            df.drop(columns=[col], inplace=True)
            dropped_columns.append({"column": col, "missing_ratio": float(missing_ratio)})
            continue

        indicator_col = f"{col}_was_missing"
        df[indicator_col] = series.isna().astype(int)
        indicator_columns.append(indicator_col)

        if pd.api.types.is_numeric_dtype(series):
            fill_value = series.median()
            df[col] = series.fillna(fill_value)
            imputation_records.append({
                "column": col,
                "strategy": "median",
                "value": None if pd.isna(fill_value) else float(fill_value)
            })
        elif pd.api.types.is_datetime64_any_dtype(series):
            fill_value = series.mode(dropna=True)
            fill_value = fill_value.iloc[0] if not fill_value.empty else series.dropna().median()
            df[col] = series.fillna(fill_value)
            value_repr = None
            if pd.notna(fill_value):
                value_repr = fill_value.isoformat() if hasattr(fill_value, "isoformat") else str(fill_value)
            imputation_records.append({
                "column": col,
                "strategy": "most_frequent",
                "value": value_repr
            })
        else:
            fill_value = series.mode(dropna=True)
            fill_value = fill_value.iloc[0] if not fill_value.empty else "unknown"
            df[col] = series.fillna(fill_value)
            imputation_records.append({"column": col, "strategy": "mode", "value": fill_value})

    if imputation_records or dropped_columns or indicator_columns:
        log.append(
            f"Missing Data Handling: imputations={imputation_records}, dropped={dropped_columns}, indicators={indicator_columns}"
        )

    if indicator_columns:
        roles = actions.get("schema_cleaning", {}).setdefault("roles", {})
        for indicator_col in indicator_columns:
            roles[indicator_col] = "missing_indicator"

    actions["missing_data"] = {
        "imputations": imputation_records,
        "dropped_columns": dropped_columns,
        "indicator_columns": indicator_columns
    }
    return df


def _auto_outlier_handling(df: pd.DataFrame, log: List[str], actions: Dict[str, Any]) -> pd.DataFrame:
    roles = actions.get("schema_cleaning", {}).get("roles", {})
    outlier_actions = []
    indicator_columns = []

    for col in df.select_dtypes(include=["number"]).columns:
        if roles.get(col) in {"id", "missing_indicator", "outlier_indicator"}:
            continue

        series = df[col]
        if series.isna().all():
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isfinite(iqr) is False:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        domain_lower = None
        domain_upper = None
        if _column_contains_keyword(col, POSITIVE_KEYWORDS):
            domain_lower = 0
        if _column_contains_keyword(col, PERCENTAGE_KEYWORDS):
            domain_lower = 0
            domain_upper = 100
        if "age" in col.lower():
            domain_lower = 0
            domain_upper = 120

        if domain_lower is not None:
            lower = max(lower, domain_lower)
        if domain_upper is not None:
            upper = min(upper, domain_upper)

        flag_name = f"{col}_outlier_flag"
        flag_series = ((series < lower) | (series > upper)).astype(int)
        if flag_series.any():
            df[flag_name] = flag_series
            indicator_columns.append(flag_name)
            roles[flag_name] = "outlier_indicator"

        clipped = series.clip(lower, upper)
        if not clipped.equals(series):
            df[col] = clipped
        outlier_actions.append({
            "column": col,
            "method": "iqr_clip",
            "lower": float(lower),
            "upper": float(upper),
            "flag_column": flag_name if flag_series.any() else None,
            "domain_constraints": {
                "lower": domain_lower,
                "upper": domain_upper
            }
        })

    if outlier_actions:
        log.append(f"Outlier Handling: {outlier_actions}")
    actions["outlier_handling"] = outlier_actions
    return df


def _auto_consistency_checks(df: pd.DataFrame, log: List[str], actions: Dict[str, Any]) -> pd.DataFrame:
    roles = actions.get("schema_cleaning", {}).get("roles", {})
    removed_rows = 0
    before_rows = len(df)
    df = df.drop_duplicates()
    removed_rows = before_rows - len(df)
    constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_columns:
        df = df.drop(columns=constant_columns)
    adjustments = []

    for col in df.select_dtypes(include=["number"]).columns:
        if roles.get(col) in {"id", "missing_indicator", "outlier_indicator"}:
            continue
        series = df[col]
        if _column_contains_keyword(col, POSITIVE_KEYWORDS):
            negatives = series < 0
            if negatives.any():
                df.loc[negatives, col] = series.clip(lower=0)
                adjustments.append({"column": col, "rule": "non_negative", "rows": int(negatives.sum())})
        if _column_contains_keyword(col, PERCENTAGE_KEYWORDS):
            bounds = ((series < 0) | (series > 100))
            if bounds.any():
                df.loc[bounds, col] = series.clip(0, 100)
                adjustments.append({"column": col, "rule": "percentage_range", "rows": int(bounds.sum())})

    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    start_cols = [col for col in date_cols if any(token in col.lower() for token in ["start", "from"])]
    end_cols = [col for col in date_cols if any(token in col.lower() for token in ["end", "to"])]
    date_corrections = []
    for start_col in start_cols:
        for end_col in end_cols:
            if start_col == end_col:
                continue
            base_start = re.sub(r"start|from", "", start_col, flags=re.IGNORECASE)
            base_end = re.sub(r"end|to", "", end_col, flags=re.IGNORECASE)
            if base_start and base_start.lower() == base_end.lower():
                invalid_mask = df[start_col] > df[end_col]
                if invalid_mask.any():
                    df.loc[invalid_mask, end_col] = df.loc[invalid_mask, start_col]
                    date_corrections.append({
                        "start": start_col,
                        "end": end_col,
                        "rows": int(invalid_mask.sum())
                    })

    actions["consistency"] = {
        "duplicates_removed": int(removed_rows),
        "constant_columns_dropped": constant_columns,
        "numeric_adjustments": adjustments,
        "date_corrections": date_corrections
    }
    if any([removed_rows, constant_columns, adjustments, date_corrections]):
        log.append(
            f"Consistency Checks: removed {removed_rows} duplicates, dropped {constant_columns}, numeric adjustments={adjustments}, date corrections={date_corrections}"
        )
    return df


def _auto_feature_engineering(df: pd.DataFrame, log: List[str], actions: Dict[str, Any]) -> pd.DataFrame:
    roles = actions.get("schema_cleaning", {}).get("roles", {})
    engineered = {
        "datetime": [],
        "skewed_numeric": [],
        "categorical_encoded": [],
        "text_features": [],
        "indicator_features": actions.get("missing_data", {}).get("indicator_columns", [])
    }

    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        engineered["datetime"].append(col)
        roles[f"{col}_year"] = "numeric_feature"
        roles[f"{col}_month"] = "numeric_feature"
        roles[f"{col}_day"] = "numeric_feature"

    skewed_cols = []
    for col in df.select_dtypes(include=["number"]).columns:
        if roles.get(col) in {"id", "missing_indicator", "outlier_indicator"}:
            continue
        if _log_skewness(df[col]):
            df[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0))
            skewed_cols.append(col)
            roles[f"{col}_log1p"] = "numeric_feature"
    engineered["skewed_numeric"] = skewed_cols

    text_cols = [col for col, role in roles.items() if role == "text" and col in df.columns]
    for col in text_cols:
        df[f"{col}_length"] = df[col].astype(str).str.len()
        df[f"{col}_word_count"] = df[col].astype(str).str.split().apply(len)
        engineered["text_features"].append(col)
        roles[f"{col}_length"] = "numeric_feature"
        roles[f"{col}_word_count"] = "numeric_feature"

    categorical_cols = [
        col for col in df.columns
        if pd.api.types.is_object_dtype(df[col]) and roles.get(col) not in {"text"}
    ]
    encoded_summary = []
    for col in categorical_cols:
        cardinality = df[col].nunique(dropna=True)
        if cardinality <= 1:
            continue
        if cardinality <= 15:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoded_summary.append({"column": col, "method": "onehot", "new_columns": list(dummies.columns)})
            for new_col in dummies.columns:
                roles[new_col] = "numeric_feature"
        else:
            df[col] = pd.factorize(df[col])[0]
            encoded_summary.append({"column": col, "method": "label"})
            roles[col] = "numeric_feature"
    engineered["categorical_encoded"] = encoded_summary

    if any([engineered["datetime"], engineered["skewed_numeric"], encoded_summary, engineered["text_features"], engineered["indicator_features"]]):
        log.append(f"Feature Engineering: created features {engineered}")
    actions["feature_engineering"] = engineered
    actions.setdefault("schema_cleaning", {}).setdefault("roles", {}).update(roles)
    return df


def _ensure_ml_ready(df: pd.DataFrame, log: List[str], actions: Dict[str, Any]) -> pd.DataFrame:
    residual = []
    if df.isna().any().any():
        for col in df.columns:
            series = df[col]
            if series.isna().mean() == 0:
                continue
            if pd.api.types.is_numeric_dtype(series):
                fill_value = series.median()
                df[col] = series.fillna(fill_value)
                residual.append({
                    "column": col,
                    "strategy": "median",
                    "value": None if pd.isna(fill_value) else float(fill_value)
                })
            elif pd.api.types.is_datetime64_any_dtype(series):
                fill_value = series.mode(dropna=True)
                fill_value = fill_value.iloc[0] if not fill_value.empty else series.dropna().median()
                df[col] = series.fillna(fill_value)
                value_repr = None
                if pd.notna(fill_value):
                    value_repr = fill_value.isoformat() if hasattr(fill_value, "isoformat") else str(fill_value)
                residual.append({
                    "column": col,
                    "strategy": "most_frequent",
                    "value": value_repr
                })
            else:
                fill_value = series.mode(dropna=True)
                fill_value = fill_value.iloc[0] if not fill_value.empty else "unknown"
                df[col] = series.fillna(fill_value)
                residual.append({"column": col, "strategy": "mode", "value": fill_value})
        if residual:
            log.append(f"ML Ready: resolved residual missing values {residual}")
    bool_cols = [col for col in df.columns if pd.api.types.is_bool_dtype(df[col])]
    for col in bool_cols:
        df[col] = df[col].astype(int)
    actions["ml_ready"] = {
        "total_features": len(df.columns),
        "rows": len(df),
        "bool_columns_encoded": bool_cols,
        "residual_fillins": residual
    }
    log.append(f"ML Ready: dataset shape {df.shape}")
    return df


def run_full_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    transformation_log: List[str] = []
    layer_actions: Dict[str, Any] = {}

    layer_actions["raw_data"] = {"rows": len(df), "columns": list(df.columns)}
    layer_actions["quality_metrics"] = _compute_quality_metrics(df)
    transformation_log.append(
        f"Raw Data: loaded dataset with {len(df)} rows and {len(df.columns)} columns"
    )

    df = _auto_schema_cleaning(df, transformation_log, layer_actions)
    df = _auto_semantic_cleaning(df, transformation_log, layer_actions)
    df = _auto_missing_data(df.copy(), transformation_log, layer_actions)
    df = _auto_outlier_handling(df, transformation_log, layer_actions)
    df = _auto_consistency_checks(df, transformation_log, layer_actions)
    df = _auto_feature_engineering(df, transformation_log, layer_actions)
    df = _ensure_ml_ready(df, transformation_log, layer_actions)
    layer_actions["post_quality_metrics"] = _compute_quality_metrics(df)

    return df, transformation_log, layer_actions

@app.post("/profile")
async def generate_profile(request: ProfileRequest):
    logger.info(f"Profile request received: file_path={request.file_path}")
    original_path = request.file_path
    file_path = download_file_if_url(original_path)
    logger.info(f"File path after URL check: {file_path}")
    
    try:
        # Read the file
        if file_path.endswith('.csv') or original_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls') or original_path.endswith('.xlsx') or original_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        logger.info(f"Loaded dataframe: {len(df)} rows, {len(df.columns)} columns")
        
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
        
        logger.info(f"Profile generated successfully: {len(profile['columns'])} columns profiled")
        return profile

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
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
            
        df, auto_log, layer_actions = run_full_pipeline(df)
        transformation_log = list(auto_log)

        rename_map = layer_actions.get("schema_cleaning", {}).get("renamed", {})
        manual_steps = [step.dict() for step in steps] if steps else []
        if rename_map and manual_steps:
            inv_map = {old: new for old, new in rename_map.items()}
            for step in manual_steps:
                column = step.get("column")
                if column in inv_map:
                    step["column"] = inv_map[column]

        manual_actions = []
        for step in manual_steps:
            col = step.get("column")
            action = step.get("action")
            method = step.get("method")
            params = step.get("params") or {}

            if action == "drop":
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
                    transformation_log.append(f"Manual: dropped column {col}")
                    manual_actions.append({"column": col, "action": action})
                continue

            if col not in df.columns:
                transformation_log.append(f"Manual: skipped {action} for missing column {col}")
                continue

            if action == "impute":
                if method == "mean":
                    val = df[col].mean()
                    df[col] = df[col].fillna(val)
                    transformation_log.append(f"Manual: imputed {col} with mean {val}")
                    manual_actions.append({"column": col, "action": action, "method": method})
                elif method == "median":
                    val = df[col].median()
                    df[col] = df[col].fillna(val)
                    transformation_log.append(f"Manual: imputed {col} with median {val}")
                    manual_actions.append({"column": col, "action": action, "method": method})
                elif method == "mode":
                    mode_val = df[col].mode(dropna=True)
                    if not mode_val.empty:
                        val = mode_val.iloc[0]
                        df[col] = df[col].fillna(val)
                        transformation_log.append(f"Manual: imputed {col} with mode {val}")
                        manual_actions.append({"column": col, "action": action, "method": method})
                elif method == "custom":
                    val = params.get("value")
                    df[col] = df[col].fillna(val)
                    transformation_log.append(f"Manual: imputed {col} with custom value {val}")
                    manual_actions.append({"column": col, "action": action, "method": method, "value": val})

            elif action == "scale":
                if method == "standard":
                    scaler = StandardScaler()
                    df[[col]] = scaler.fit_transform(df[[col]])
                    transformation_log.append(f"Manual: standard scaled {col}")
                    manual_actions.append({"column": col, "action": action, "method": method})
                elif method == "minmax":
                    scaler = MinMaxScaler()
                    df[[col]] = scaler.fit_transform(df[[col]])
                    transformation_log.append(f"Manual: minmax scaled {col}")
                    manual_actions.append({"column": col, "action": action, "method": method})

            elif action == "encode":
                if method == "onehot":
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    transformation_log.append(f"Manual: one-hot encoded {col}")
                    manual_actions.append({
                        "column": col,
                        "action": action,
                        "method": method,
                        "new_columns": list(dummies.columns)
                    })
                elif method == "label":
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    transformation_log.append(f"Manual: label encoded {col}")
                    manual_actions.append({"column": col, "action": action, "method": method})

        if manual_actions:
            layer_actions["manual_steps"] = manual_actions

        if manual_actions:
            df = _ensure_ml_ready(df, transformation_log, layer_actions)
                    
        # Save Cleaned File
        base, ext = os.path.splitext(file_path)
        cleaned_path = f"{base}_cleaned{ext}"
        
        if ext == '.csv':
            df.to_csv(cleaned_path, index=False)
        else:
            df.to_excel(cleaned_path, index=False)
        
        # Generate Python reproducible code
        python_code = generate_pipeline_code(file_path, layer_actions, manual_steps)
            
        return {
            "cleaned_file_path": cleaned_path,
            "transformation_log": transformation_log,
            "preview": df.replace({np.nan: None}).head(20).to_dict(orient='records'),
            "columns": list(df.columns),
            "python_code": python_code,
            "layer_summary": layer_actions
        }

    except Exception as e:
        print(f"Error preparing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_pipeline_code(file_path: str, actions: Dict[str, Any], manual_steps: List[Dict[str, Any]]) -> str:
    """Generate reproducible Python code for the automatic and manual pipeline layers."""
    code_lines = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder",
        "",
        "# Load dataset",
        f"df = pd.read_csv('{os.path.basename(file_path)}')",
        "",
        "# ---------- Schema Cleaning ----------"
    ]

    schema = actions.get("schema_cleaning", {})
    rename_map = schema.get("renamed", {})
    if rename_map:
        code_lines.append(f"df = df.rename(columns={json.dumps(rename_map)})")
    for cast in schema.get("type_conversions", []):
        col = cast.get("column")
        target = cast.get("target_type")
        if target == "numeric":
            code_lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')")
        elif target == "datetime":
            code_lines.append(
                f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')"
            )
        elif target == "boolean":
            code_lines.append(
                f"df['{col}'] = df['{col}'].astype(str).str.lower().str.strip().isin(['true','yes','y','1']).astype(int)"
            )
    roles = schema.get("roles", {})

    code_lines.append("\n# ---------- Semantic Cleaning ----------")
    semantic = actions.get("semantic_cleaning", {})
    for col in semantic.get("dates", []):
        code_lines.append(f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce').dt.tz_localize(None)")
    for col in semantic.get("categorical_standardized", []):
        code_lines.append(
            f"df['{col}'] = df['{col}'].astype(str).str.strip().str.replace(r'\\s+', ' ', regex=True)"
        )
    for col in semantic.get("missing_tokens", []):
        tokens = sorted(list(MISSING_TOKENS))
        code_lines.append(
            f"df['{col}'] = df['{col}'].replace({{token: np.nan for token in {tokens}}})"
        )
    for col, mapping_key in semantic.get("canonical_mappings", {}).items():
        if mapping_key == "gender":
            mapping = SEMANTIC_MAPPINGS.get("gender", {})
            code_lines.append(
                f"df['{col}'] = df['{col}'].str.lower().map({json.dumps(mapping)}).fillna(df['{col}'].str.lower())"
            )
        elif mapping_key == "boolean":
            mapping = SEMANTIC_MAPPINGS.get("boolean", {})
            code_lines.append(
                f"df['{col}'] = df['{col}'].str.lower().map({json.dumps(mapping)}).fillna(df['{col}'].str.lower())"
            )

    code_lines.append("\n# ---------- Missing Data Handling ----------")
    missing_actions = actions.get("missing_data", {})
    for dropped in missing_actions.get("dropped_columns", []):
        drop_col = dropped.get("column")
        code_lines.append(f"df.drop(columns=['{drop_col}'], inplace=True)")
    for indicator in missing_actions.get("indicator_columns", []):
        source_col = indicator.replace("_was_missing", "")
        code_lines.append(
            f"df['{indicator}'] = df['{source_col}'].isna().astype(int)"
        )
    for record in missing_actions.get("imputations", []):
        col = record.get("column")
        strategy = record.get("strategy")
        value = record.get("value")
        if strategy in {"median", "mean"} and value is not None:
            code_lines.append(f"df['{col}'].fillna({value}, inplace=True)")
        elif strategy == "mode":
            code_lines.append(f"df['{col}'].fillna({json.dumps(value)}, inplace=True)")
        elif strategy == "most_frequent":
            if value is not None:
                code_lines.append(f"df['{col}'].fillna(pd.to_datetime('{value}'), inplace=True)")

    code_lines.append("\n# ---------- Outlier Handling ----------")
    for record in actions.get("outlier_handling", []):
        col = record.get("column")
        lower = record.get("lower")
        upper = record.get("upper")
        flag = record.get("flag_column")
        if flag:
            code_lines.append(
                f"df['{flag}'] = ((df['{col}'] < {lower}) | (df['{col}'] > {upper})).astype(int)"
            )
        code_lines.append(f"df['{col}'] = df['{col}'].clip({lower}, {upper})")

    code_lines.append("\n# ---------- Consistency Checks ----------")
    consistency = actions.get("consistency", {})
    code_lines.append("df = df.drop_duplicates()")
    dropped_cols = consistency.get("constant_columns_dropped", [])
    if dropped_cols:
        code_lines.append(f"df.drop(columns={dropped_cols}, inplace=True)")
    for adj in consistency.get("numeric_adjustments", []):
        col = adj.get("column")
        if adj.get("rule") == "non_negative":
            code_lines.append(f"df['{col}'] = df['{col}'].clip(lower=0)")
        elif adj.get("rule") == "percentage_range":
            code_lines.append(f"df['{col}'] = df['{col}'].clip(0, 100)")
    for corr in consistency.get("date_corrections", []):
        start = corr.get("start")
        end = corr.get("end")
        code_lines.append(
            f"mask = df['{start}'] > df['{end}']"
        )
        code_lines.append(
            f"df.loc[mask, '{end}'] = df.loc[mask, '{start}']"
        )

    code_lines.append("\n# ---------- Feature Engineering ----------")
    features = actions.get("feature_engineering", {})
    for col in features.get("datetime", []):
        code_lines.append(f"df['{col}_year'] = df['{col}'].dt.year")
        code_lines.append(f"df['{col}_month'] = df['{col}'].dt.month")
        code_lines.append(f"df['{col}_day'] = df['{col}'].dt.day")
    for col in features.get("skewed_numeric", []):
        code_lines.append(f"df['{col}_log1p'] = np.log1p(df['{col}'].clip(lower=0))")
    for col in features.get("text_features", []):
        code_lines.append(f"df['{col}_length'] = df['{col}'].astype(str).str.len()")
        code_lines.append(f"df['{col}_word_count'] = df['{col}'].astype(str).str.split().apply(len)")
    for record in features.get("categorical_encoded", []):
        col = record.get("column")
        method = record.get("method")
        if method == "onehot":
            new_cols = record.get("new_columns", [])
            code_lines.append(f"dummies = pd.get_dummies(df['{col}'], prefix='{col}', dummy_na=False)")
            code_lines.append(f"df = pd.concat([df.drop(columns=['{col}']), dummies], axis=1)")
        elif method == "label":
            code_lines.append(f"df['{col}'] = pd.factorize(df['{col}'])[0]")

    if manual_steps:
        code_lines.append("\n# ---------- Manual Adjustments ----------")
    for step in manual_steps:
        col = step.get("column")
        action = step.get("action")
        method = step.get("method")
        if action == "drop":
            code_lines.append(f"df.drop(columns=['{col}'], inplace=True)")
        elif action == "impute":
            if method == "mean":
                code_lines.append(f"df['{col}'].fillna(df['{col}'].mean(), inplace=True)")
            elif method == "median":
                code_lines.append(f"df['{col}'].fillna(df['{col}'].median(), inplace=True)")
            elif method == "mode":
                code_lines.append(f"df['{col}'].fillna(df['{col}'].mode(dropna=True)[0], inplace=True)")
            elif method == "custom":
                value = step.get("params", {}).get("value")
                code_lines.append(f"df['{col}'].fillna({json.dumps(value)}, inplace=True)")
        elif action == "scale":
            if method == "standard":
                code_lines.append("scaler = StandardScaler()")
                code_lines.append(f"df[['{col}']] = scaler.fit_transform(df[['{col}']])")
            elif method == "minmax":
                code_lines.append("scaler = MinMaxScaler()")
                code_lines.append(f"df[['{col}']] = scaler.fit_transform(df[['{col}']])")
        elif action == "encode":
            if method == "onehot":
                code_lines.append(f"dummies = pd.get_dummies(df['{col}'], prefix='{col}')")
                code_lines.append(f"df = pd.concat([df.drop(columns=['{col}']), dummies], axis=1)")
            elif method == "label":
                code_lines.append("le = LabelEncoder()")
                code_lines.append(f"df['{col}'] = le.fit_transform(df['{col}'].astype(str))")

    code_lines.append("\n# ---------- Final ML Ready Checks ----------")
    ml_ready = actions.get("ml_ready", {})
    for record in ml_ready.get("residual_fillins", []):
        col = record.get("column")
        strategy = record.get("strategy")
        value = record.get("value")
        if strategy == "median" and value is not None:
            code_lines.append(f"df['{col}'].fillna({value}, inplace=True)")
        elif strategy == "mode":
            code_lines.append(f"df['{col}'].fillna({json.dumps(value)}, inplace=True)")
        elif strategy == "most_frequent" and value is not None:
            code_lines.append(f"df['{col}'].fillna(pd.to_datetime('{value}'), inplace=True)")
    for col in ml_ready.get("bool_columns_encoded", []):
        code_lines.append(f"df['{col}'] = df['{col}'].astype(int)")

    quality = actions.get("quality_metrics")
    if quality:
        code_lines.append("# Optional: inspect raw data quality metrics")
        code_lines.append("# quality_metrics = {\"rows\": %s, \"columns\": %s}" % (quality.get('rows'), quality.get('columns')))
    post_quality = actions.get("post_quality_metrics")
    if post_quality:
        code_lines.append("# cleaned_quality_metrics = {\"rows\": %s, \"columns\": %s}" % (post_quality.get('rows'), post_quality.get('columns')))

    code_lines.append("")
    code_lines.append("# Save cleaned dataset")
    code_lines.append("df.to_csv('cleaned_dataset.csv', index=False)")

    return "\n".join(code_lines)


# ========== AUTOKLEAN AUTO-CLEAN ENDPOINT ==========

class AutoCleanRequest(BaseModel):
    file_path: str
    split_ratio: Optional[float] = 0.8
    k_folds: Optional[int] = 5
    epochs: Optional[int] = 100
    remove_outliers: Optional[bool] = True
    impute_missing: Optional[bool] = True
    normalize_features: Optional[bool] = True


@app.post("/auto-clean")
async def auto_clean_dataset(request: AutoCleanRequest):
    """
    AutoKlean endpoint: Automatically clean dataset with configurable options.
    - Impute missing values using KNN or median
    - Remove outliers using IQR or Isolation Forest
    - Normalize features using MinMax scaler
    - Split data for train/test
    """
    logger.info(f"Auto-clean request received: file_path={request.file_path}")
    logger.info(f"Options: split_ratio={request.split_ratio}, remove_outliers={request.remove_outliers}, impute_missing={request.impute_missing}, normalize_features={request.normalize_features}")
    
    original_path = request.file_path
    file_path = download_file_if_url(original_path)
    logger.info(f"File path after URL check: {file_path}")
    
    try:
        # Load Data
        if file_path.endswith('.csv') or original_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls') or original_path.endswith('.xlsx') or original_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        transformation_log = []
        original_rows = len(df)
        original_cols = len(df.columns)
        transformation_log.append(f"Loaded dataset: {original_rows} rows, {original_cols} columns")
        
        # Run the standard pipeline first
        df, auto_log, layer_actions = run_full_pipeline(df)
        transformation_log.extend(auto_log)
        
        # Additional processing based on AutoKlean config
        
        # 1. Impute missing values
        if request.impute_missing:
            for col in df.columns:
                if df[col].isna().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Use median for numeric
                        median_val = df[col].median()
                        if pd.notna(median_val):
                            df[col] = df[col].fillna(median_val)
                            transformation_log.append(f"Imputed {col} with median: {median_val:.2f}")
                    else:
                        # Use mode for categorical
                        mode_series = df[col].mode()
                        mode_val = mode_series.iloc[0] if len(mode_series) > 0 else "unknown"
                        df[col] = df[col].fillna(mode_val)
                        transformation_log.append(f"Imputed {col} with mode: {mode_val}")
        
        # 2. Handle outliers (using IQR clipping for numeric columns)
        # Note: We clip values instead of removing rows to preserve data integrity
        if request.remove_outliers:
            outliers_clipped = 0
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0 or not np.isfinite(IQR):
                    continue  # Skip columns with no variance
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Count outliers before clipping
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_clipped += outlier_mask.sum()
                # Clip values instead of removing rows
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            if outliers_clipped > 0:
                transformation_log.append(f"Clipped {outliers_clipped} outlier values using IQR method")
        
        # 3. Normalize features (MinMax scaling for numeric columns)
        if request.normalize_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    transformation_log.append(f"Normalized {col} using MinMax scaling")
        
        transformation_log.append(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Save Cleaned File
        base, ext = os.path.splitext(file_path)
        cleaned_path = f"{base}_cleaned{ext}"
        
        if ext == '.csv' or not ext:
            df.to_csv(cleaned_path if ext else f"{base}_cleaned.csv", index=False)
            if not ext:
                cleaned_path = f"{base}_cleaned.csv"
        else:
            df.to_excel(cleaned_path, index=False)
            
        # 4. Train/Test Split
        train_path = None
        test_path = None
        
        # Safety check: ensure we have enough rows for splitting
        if len(df) < 2:
            raise HTTPException(status_code=400, detail="Dataset has too few rows after processing. Please check your data quality or disable some cleaning options.")
        
        if request.split_ratio and request.split_ratio > 0:
            try:
                train_size = request.split_ratio / 100.0
                # Ensure valid range
                train_size = max(0.1, min(0.95, train_size))
                
                train_df, test_df = train_test_split(df, train_size=train_size, random_state=42)
                
                # Create a directory for the split files
                timestamp = int(datetime.now().timestamp() * 1000)
                split_dir = os.path.join(os.path.dirname(file_path), f"{timestamp}_clean_and_split")
                os.makedirs(split_dir, exist_ok=True)
                
                train_path = os.path.join(split_dir, "train.csv")
                test_path = os.path.join(split_dir, "test.csv")
                
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)
                
                transformation_log.append(f"Split dataset into Train ({len(train_df)} rows) and Test ({len(test_df)} rows)")
            except Exception as e:
                transformation_log.append(f"Error during train/test split: {str(e)}")
        
        # Generate simple Python code
        python_code = f'''import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('{os.path.basename(file_path)}')

# AutoKlean Configuration
split_ratio = {request.split_ratio}
k_folds = {request.k_folds}
remove_outliers = {request.remove_outliers}
impute_missing = {request.impute_missing}
normalize_features = {request.normalize_features}

# Impute missing values
if impute_missing:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Remove outliers using IQR
if remove_outliers:
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

# Normalize features
if normalize_features:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Save cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)
'''
        
        # Convert cleaned dataframe to CSV content for transfer
        import io
        import base64
        
        cleaned_csv_buffer = io.StringIO()
        df.to_csv(cleaned_csv_buffer, index=False)
        cleaned_csv_content = base64.b64encode(cleaned_csv_buffer.getvalue().encode()).decode()
        
        # Also encode train/test if they exist
        train_csv_content = None
        test_csv_content = None
        if train_path and test_path:
            train_buffer = io.StringIO()
            test_buffer = io.StringIO()
            train_df.to_csv(train_buffer, index=False)
            test_df.to_csv(test_buffer, index=False)
            train_csv_content = base64.b64encode(train_buffer.getvalue().encode()).decode()
            test_csv_content = base64.b64encode(test_buffer.getvalue().encode()).decode()
        
        return {
            "cleaned_file_path": cleaned_path,
            "cleaned_csv_content": cleaned_csv_content,
            "train_csv_content": train_csv_content,
            "test_csv_content": test_csv_content,
            "train_file_path": train_path,
            "test_file_path": test_path,
            "transformation_log": transformation_log,
            "preview": df.replace({np.nan: None}).head(20).to_dict(orient='records'),
            "columns": list(df.columns),
            "python_code": python_code,
            "config": {
                "split_ratio": request.split_ratio,
                "k_folds": request.k_folds,
                "epochs": request.epochs,
                "remove_outliers": request.remove_outliers,
                "impute_missing": request.impute_missing,
                "normalize_features": request.normalize_features
            },
            "stats": {
                "original_rows": original_rows,
                "final_rows": len(df),
                "original_cols": original_cols,
                "final_cols": len(df.columns)
            }
        }
    
    except Exception as e:
        logger.error(f"Error in auto-clean: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== ADVANCED PIPELINE OPERATIONS ==========

class PipelineRequest(BaseModel):
    file_path: str
    operation_type: str  # clean_only, split, clean_and_split, cross_validation, clean_and_cv
    test_size: Optional[float] = 0.2
    n_folds: Optional[int] = 5
    shuffle: Optional[bool] = True
    random_state: Optional[int] = 42
    cleaning_pipeline: Optional[Dict[str, Any]] = None


def apply_cleaning_pipeline(df: pd.DataFrame, steps: List[Dict[str, Any]], log: List[str]) -> pd.DataFrame:
    """Apply cleaning transformations to dataframe"""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    
    for step in steps:
        column = step.get('column')
        action = step.get('action')
        method = step.get('method')
        
        if not column or column not in df.columns:
            continue
            
        if action == 'drop':
            df = df.drop(columns=[column])
            log.append(f"Dropped column: {column}")
            
        elif action == 'impute':
            if method == 'mean':
                value = df[column].mean()
                df[column].fillna(value, inplace=True)
                log.append(f"Imputed {column} with mean: {value:.2f}")
            elif method == 'median':
                value = df[column].median()
                df[column].fillna(value, inplace=True)
                log.append(f"Imputed {column} with median: {value:.2f}")
            elif method == 'mode':
                value = df[column].mode()[0] if not df[column].mode().empty else df[column].iloc[0]
                df[column].fillna(value, inplace=True)
                log.append(f"Imputed {column} with mode: {value}")
                
        elif action == 'scale':
            if method == 'standard':
                scaler = StandardScaler()
                df[[column]] = scaler.fit_transform(df[[column]])
                log.append(f"Applied StandardScaler to {column}")
            elif method == 'minmax':
                scaler = MinMaxScaler()
                df[[column]] = scaler.fit_transform(df[[column]])
                log.append(f"Applied MinMaxScaler to {column}")
                
        elif action == 'encode':
            if method == 'label':
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                log.append(f"Applied LabelEncoder to {column}")
            elif method == 'onehot':
                dummies = pd.get_dummies(df[column], prefix=column)
                df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
                log.append(f"Applied OneHotEncoder to {column}")
    
    return df


def train_test_split_operation(df: pd.DataFrame, test_size: float, shuffle: bool, 
                                random_state: int, output_dir: str) -> Tuple[Dict[str, Any], List[str]]:
    """Split dataset into train and test sets"""
    from sklearn.model_selection import train_test_split
    
    log = []
    
    # Validate test_size
    if test_size < 0.1 or test_size > 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")
    
    if len(df) < 10:
        raise ValueError("Dataset too small for splitting (minimum 10 rows required)")
    
    # Perform split
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        shuffle=shuffle, 
        random_state=random_state
    )
    
    # Save outputs
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    log.append(f"Split dataset: {len(train_df)} train rows, {len(test_df)} test rows")
    log.append(f"Train/Test ratio: {(1-test_size)*100:.0f}%/{test_size*100:.0f}%")
    
    output_files = [
        {
            'name': 'train.csv',
            'path': f'/uploads/{os.path.basename(output_dir)}/train.csv',
            'rows': len(train_df),
            'columns': len(train_df.columns)
        },
        {
            'name': 'test.csv',
            'path': f'/uploads/{os.path.basename(output_dir)}/test.csv',
            'rows': len(test_df),
            'columns': len(test_df.columns)
        }
    ]
    
    metadata = {
        'split_type': 'train_test',
        'test_size': test_size,
        'rows_per_output': {
            'train': len(train_df),
            'test': len(test_df)
        }
    }
    
    return {'output_files': output_files, 'metadata': metadata}, log


def cross_validation_split_operation(df: pd.DataFrame, n_folds: int, shuffle: bool, 
                                     random_state: int, output_dir: str) -> Tuple[Dict[str, Any], List[str]]:
    """Split dataset into K folds for cross-validation"""
    from sklearn.model_selection import KFold
    
    log = []
    
    # Validate n_folds
    if n_folds < 2 or n_folds > 10:
        raise ValueError("n_folds must be between 2 and 10")
    
    if len(df) < n_folds * 2:
        raise ValueError(f"Dataset too small for {n_folds}-fold CV (minimum {n_folds * 2} rows required)")
    
    # Perform K-Fold split
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
    
    output_files = []
    fold_num = 1
    
    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_path = os.path.join(output_dir, f'fold_{fold_num}_train.csv')
        val_path = os.path.join(output_dir, f'fold_{fold_num}_val.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        output_files.extend([
            {
                'name': f'fold_{fold_num}_train.csv',
                'path': f'/uploads/{os.path.basename(output_dir)}/fold_{fold_num}_train.csv',
                'rows': len(train_df),
                'columns': len(train_df.columns)
            },
            {
                'name': f'fold_{fold_num}_val.csv',
                'path': f'/uploads/{os.path.basename(output_dir)}/fold_{fold_num}_val.csv',
                'rows': len(val_df),
                'columns': len(val_df.columns)
            }
        ])
        
        log.append(f"Fold {fold_num}: {len(train_df)} train, {len(val_df)} validation rows")
        fold_num += 1
    
    metadata = {
        'split_type': 'kfold',
        'n_folds': n_folds,
        'rows_per_fold': {
            f'fold_{i+1}': {
                'train': output_files[i*2]['rows'],
                'val': output_files[i*2+1]['rows']
            } for i in range(n_folds)
        }
    }
    
    return {'output_files': output_files, 'metadata': metadata}, log


def _load_dataframe(file_path: str) -> pd.DataFrame:
    """Load dataframe from CSV or Excel file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")


@app.post("/run-pipeline")
async def run_pipeline(req: PipelineRequest):
    """
    Advanced pipeline operations: clean, split, or combine operations
    """
    try:
        # Load dataset
        df = _load_dataframe(req.file_path)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Could not load or empty dataframe")
        
        original_rows = len(df)
        transformation_log = []
        
        # Create output directory
        timestamp = int(datetime.now().timestamp() * 1000)
        output_dir_name = f"{timestamp}_{req.operation_type}"
        output_dir = os.path.join(os.path.dirname(req.file_path), output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        transformation_log.append(f"Starting pipeline: {req.operation_type}")
        transformation_log.append(f"Original dataset: {original_rows} rows, {len(df.columns)} columns")
        
        # Execute operation based on type
        if req.operation_type == 'clean_only':
            # Apply cleaning pipeline
            if req.cleaning_pipeline and req.cleaning_pipeline.get('steps'):
                df = apply_cleaning_pipeline(df, req.cleaning_pipeline['steps'], transformation_log)
            
            cleaned_path = os.path.join(output_dir, 'cleaned.csv')
            df.to_csv(cleaned_path, index=False)
            
            output_files = [{
                'name': 'cleaned.csv',
                'path': f'/uploads/{output_dir_name}/cleaned.csv',
                'rows': len(df),
                'columns': len(df.columns)
            }]
            
            metadata = {
                'rows_before': original_rows,
                'rows_after_cleaning': len(df),
                'operation': 'clean_only'
            }
            
        elif req.operation_type == 'split':
            # Split without cleaning
            result, split_log = train_test_split_operation(
                df, req.test_size, req.shuffle, req.random_state, output_dir
            )
            transformation_log.extend(split_log)
            
            output_files = result['output_files']
            metadata = {
                'rows_before': original_rows,
                'operation': 'split',
                **result['metadata']
            }
            
        elif req.operation_type == 'clean_and_split':
            # Clean then split
            if req.cleaning_pipeline and req.cleaning_pipeline.get('steps'):
                df = apply_cleaning_pipeline(df, req.cleaning_pipeline['steps'], transformation_log)
            
            rows_after_cleaning = len(df)
            
            result, split_log = train_test_split_operation(
                df, req.test_size, req.shuffle, req.random_state, output_dir
            )
            transformation_log.extend(split_log)
            
            output_files = result['output_files']
            metadata = {
                'rows_before': original_rows,
                'rows_after_cleaning': rows_after_cleaning,
                'operation': 'clean_and_split',
                **result['metadata']
            }
            
        elif req.operation_type == 'cross_validation':
            # Cross-validation without cleaning
            result, cv_log = cross_validation_split_operation(
                df, req.n_folds, req.shuffle, req.random_state, output_dir
            )
            transformation_log.extend(cv_log)
            
            output_files = result['output_files']
            metadata = {
                'rows_before': original_rows,
                'operation': 'cross_validation',
                **result['metadata']
            }
            
        elif req.operation_type == 'clean_and_cv':
            # Clean then cross-validation
            if req.cleaning_pipeline and req.cleaning_pipeline.get('steps'):
                df = apply_cleaning_pipeline(df, req.cleaning_pipeline['steps'], transformation_log)
            
            rows_after_cleaning = len(df)
            
            result, cv_log = cross_validation_split_operation(
                df, req.n_folds, req.shuffle, req.random_state, output_dir
            )
            transformation_log.extend(cv_log)
            
            output_files = result['output_files']
            metadata = {
                'rows_before': original_rows,
                'rows_after_cleaning': rows_after_cleaning,
                'operation': 'clean_and_cv',
                **result['metadata']
            }
        else:
            raise HTTPException(status_code=400, detail=f"Invalid operation_type: {req.operation_type}")
        
        transformation_log.append(f"Pipeline completed successfully")
        
        # Generate Python code
        python_code = _generate_pipeline_code(req, transformation_log)
        
        return {
            'output_files': output_files,
            'metadata': metadata,
            'transformation_log': transformation_log,
            'python_code': python_code,
            'output_dir': output_dir
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _generate_pipeline_code(req: PipelineRequest, log: List[str]) -> str:
    """Generate Python code for the pipeline"""
    lines = [
        "import pandas as pd",
        "from sklearn.model_selection import train_test_split, KFold",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder",
        "from sklearn.impute import SimpleImputer",
        "",
        "# Load dataset",
        f"df = pd.read_csv('{os.path.basename(req.file_path)}')",
        f"print(f'Original dataset: {{len(df)}} rows, {{len(df.columns)}} columns')",
        ""
    ]
    
    # Add cleaning steps if applicable
    if req.cleaning_pipeline and req.cleaning_pipeline.get('steps'):
        lines.append("# Cleaning pipeline")
        for step in req.cleaning_pipeline['steps']:
            col = step.get('column')
            action = step.get('action')
            method = step.get('method')
            
            if action == 'impute' and method == 'mean':
                lines.append(f"df['{col}'].fillna(df['{col}'].mean(), inplace=True)")
            elif action == 'impute' and method == 'median':
                lines.append(f"df['{col}'].fillna(df['{col}'].median(), inplace=True)")
            elif action == 'impute' and method == 'mode':
                lines.append(f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)")
            elif action == 'scale' and method == 'standard':
                lines.append(f"scaler = StandardScaler()")
                lines.append(f"df[['{col}']] = scaler.fit_transform(df[['{col}']])")
            elif action == 'encode' and method == 'label':
                lines.append(f"le = LabelEncoder()")
                lines.append(f"df['{col}'] = le.fit_transform(df['{col}'].astype(str))")
        lines.append("")
    
    # Add split logic
    if 'split' in req.operation_type:
        lines.extend([
            "# Train/Test split",
            f"train_df, test_df = train_test_split(df, test_size={req.test_size}, shuffle={req.shuffle}, random_state={req.random_state})",
            "train_df.to_csv('train.csv', index=False)",
            "test_df.to_csv('test.csv', index=False)",
            "print(f'Train: {{len(train_df)}} rows | Test: {{len(test_df)}} rows')"
        ])
    elif 'cv' in req.operation_type:
        lines.extend([
            "# K-Fold Cross-Validation",
            f"kf = KFold(n_splits={req.n_folds}, shuffle={req.shuffle}, random_state={req.random_state if req.shuffle else None})",
            "fold_num = 1",
            "for train_idx, val_idx in kf.split(df):",
            "    train_df = df.iloc[train_idx]",
            "    val_df = df.iloc[val_idx]",
            "    train_df.to_csv(f'fold_{fold_num}_train.csv', index=False)",
            "    val_df.to_csv(f'fold_{fold_num}_val.csv', index=False)",
            "    print(f'Fold {fold_num}: {len(train_df)} train, {len(val_df)} val')",
            "    fold_num += 1"
        ])
    elif req.operation_type == 'clean_only':
        lines.append("df.to_csv('cleaned.csv', index=False)")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
