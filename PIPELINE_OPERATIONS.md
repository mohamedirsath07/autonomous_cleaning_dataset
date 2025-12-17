# Advanced Dataset Pipeline Operations

## Overview

This document describes the advanced dataset operations feature that extends the Autonomous Data Preparation Engine with cleaning, train-test split, and cross-validation capabilities.

## Features

### 5 Pipeline Operations

#### 1️⃣ Clean Dataset Only
- **Description**: Apply cleaning pipeline without splitting
- **Output**: Single cleaned dataset
- **Use Case**: When you only need to clean data without creating train/test splits

#### 2️⃣ Split Dataset (Train/Test)
- **Description**: Split raw dataset into training and test sets
- **Inputs**:
  - Test size (10% - 40%)
  - Random seed (optional)
- **Output**: 
  - `train.csv`
  - `test.csv`
- **Use Case**: Quick train/test split without cleaning

#### 3️⃣ Clean Dataset + Train/Test Split
- **Description**: Clean dataset first, then split into train and test
- **Inputs**:
  - Cleaning options (imputation, scaling, encoding)
  - Test size (10% - 40%)
  - Random seed (optional)
- **Output**: 
  - `train.csv` (cleaned and split)
  - `test.csv` (cleaned and split)
- **Use Case**: Most common ML workflow - clean then split

#### 4️⃣ Cross-Validation Split
- **Description**: Split dataset into K folds for cross-validation
- **Inputs**:
  - Number of folds K (2-10)
  - Shuffle (true/false)
  - Random seed (optional)
- **Output**:
  - `fold_1_train.csv`, `fold_1_val.csv`
  - `fold_2_train.csv`, `fold_2_val.csv`
  - ... up to K folds
- **Use Case**: Robust model evaluation without cleaning

#### 5️⃣ Clean Dataset + Cross-Validation
- **Description**: Clean dataset first, then perform K-fold cross-validation
- **Inputs**:
  - Cleaning options
  - Number of folds K (2-10)
  - Shuffle option
  - Random seed (optional)
- **Output**: Cleaned K-fold train/validation datasets
- **Use Case**: Robust model evaluation with cleaned data

## Frontend (React)

### UI Components

1. **Operation Selector**
   - Radio buttons for 5 operation types
   - Clear descriptions for each operation
   - Visual indication of selected operation

2. **Conditional Inputs**
   - Test size slider (10%-40%) - shown for split operations
   - Number of folds input (2-10) - shown for CV operations
   - Shuffle checkbox - shown for split/CV operations
   - Random seed input - always shown

3. **Execution Button**
   - "Run Data Pipeline" button
   - Loading state during execution

4. **Results Section**
   - Download links for each output file
   - File metadata (rows × columns)
   - Pipeline summary
   - Transformation log
   - Generated Python code

### State Management

```javascript
const [operationType, setOperationType] = useState('clean_only');
const [testSize, setTestSize] = useState(0.2);
const [nFolds, setNFolds] = useState(5);
const [shuffle, setShuffle] = useState(true);
const [randomState, setRandomState] = useState(42);
const [pipelineResult, setPipelineResult] = useState(null);
```

## Backend API (Node.js)

### Endpoint

**POST** `/api/datasets/:id/run-pipeline`

### Request Payload

```json
{
  "operation_type": "clean_and_split",
  "test_size": 0.2,
  "n_folds": 5,
  "shuffle": true,
  "random_state": 42,
  "cleaning_pipeline": {
    "steps": [
      {
        "column": "age",
        "action": "impute",
        "method": "median"
      },
      {
        "column": "salary",
        "action": "scale",
        "method": "standard"
      }
    ]
  }
}
```

### Response

```json
{
  "output_files": [
    {
      "name": "train.csv",
      "path": "/uploads/1234567890_clean_and_split/train.csv",
      "rows": 8000,
      "columns": 15
    },
    {
      "name": "test.csv",
      "path": "/uploads/1234567890_clean_and_split/test.csv",
      "rows": 2000,
      "columns": 15
    }
  ],
  "metadata": {
    "rows_before": 10000,
    "rows_after_cleaning": 10000,
    "operation": "clean_and_split",
    "split_type": "train_test",
    "test_size": 0.2,
    "rows_per_output": {
      "train": 8000,
      "test": 2000
    }
  },
  "transformation_log": [
    "Starting pipeline: clean_and_split",
    "Original dataset: 10000 rows, 15 columns",
    "Imputed age with median: 35.00",
    "Applied StandardScaler to salary",
    "Split dataset: 8000 train rows, 2000 test rows"
  ],
  "python_code": "...",
  "run_id": "abc123..."
}
```

### Validation Rules

- `operation_type` must be one of: `clean_only`, `split`, `clean_and_split`, `cross_validation`, `clean_and_cv`
- `test_size` must be between 0.1 and 0.4
- `n_folds` must be between 2 and 10
- Dataset must have sufficient rows for splitting

## Python ML Service (FastAPI)

### Main Endpoint

**POST** `/run-pipeline`

### Core Functions

#### 1. `apply_cleaning_pipeline(df, steps, log)`
Applies cleaning transformations:
- Missing value imputation (mean, median, mode)
- Outlier handling
- Scaling (standard, minmax)
- Encoding (label, onehot)
- Column dropping

#### 2. `train_test_split_operation(df, test_size, shuffle, random_state, output_dir)`
Performs train/test split using scikit-learn:
- Validates test_size and dataset size
- Splits data maintaining distribution
- Saves train.csv and test.csv
- Returns file metadata

#### 3. `cross_validation_split_operation(df, n_folds, shuffle, random_state, output_dir)`
Performs K-fold cross-validation:
- Validates n_folds and dataset size
- Creates K stratified folds
- Saves fold_i_train.csv and fold_i_val.csv for each fold
- Returns file metadata for all folds

### Output Directory Structure

```
/uploads/
 ├── 1765434567890_clean_only/
 │   └── cleaned.csv
 ├── 1765434567891_split/
 │   ├── train.csv
 │   └── test.csv
 ├── 1765434567892_clean_and_split/
 │   ├── train.csv
 │   └── test.csv
 ├── 1765434567893_cross_validation/
 │   ├── fold_1_train.csv
 │   ├── fold_1_val.csv
 │   ├── fold_2_train.csv
 │   ├── fold_2_val.csv
 │   └── ... (up to K folds)
 └── 1765434567894_clean_and_cv/
     ├── fold_1_train.csv
     ├── fold_1_val.csv
     └── ... (up to K folds)
```

### Transformation Log Structure

```json
{
  "transformation_log": [
    "Starting pipeline: clean_and_split",
    "Original dataset: 10000 rows, 15 columns",
    "Imputed age with median: 35.00",
    "Applied StandardScaler to salary",
    "Split dataset: 8000 train rows, 2000 test rows",
    "Train/Test ratio: 80%/20%",
    "Pipeline completed successfully"
  ]
}
```

## Validation & Error Handling

### Dataset Validation
- ✅ Minimum 10 rows for train/test split
- ✅ Minimum K×2 rows for K-fold CV
- ✅ Non-empty dataframe

### Input Validation
- ✅ Test size between 0.1 and 0.4
- ✅ Number of folds between 2 and 10
- ✅ Valid operation type
- ✅ Valid cleaning steps

### Edge Cases Handled
- Dataset too small for splitting
- Invalid column names in cleaning steps
- Missing or corrupted files
- Invalid parameters

## Generated Python Code

The system generates reproducible Python code for each pipeline:

```python
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('dataset.csv')
print(f'Original dataset: {len(df)} rows, {len(df.columns)} columns')

# Cleaning pipeline
df['age'].fillna(df['age'].median(), inplace=True)
scaler = StandardScaler()
df[['salary']] = scaler.fit_transform(df[['salary']])

# Train/Test split
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
print(f'Train: {len(train_df)} rows | Test: {len(test_df)} rows')
```

## Usage Example

### 1. Upload Dataset
Upload your CSV/XLSX file through the UI

### 2. Configure Pipeline
- Select operation type (e.g., "Clean + Train/Test Split")
- Configure cleaning steps in the profile builder
- Set split parameters (test size: 20%, random seed: 42)

### 3. Run Pipeline
Click "Run Data Pipeline" button

### 4. Download Results
- Download individual files (train.csv, test.csv)
- View transformation log
- Copy generated Python code
- Access pipeline metadata

## Database Schema

### PipelineRun Model

```javascript
{
  dataset_id: ObjectId,
  pipeline_config: {
    operation_type: String,
    test_size: Number,
    n_folds: Number,
    shuffle: Boolean,
    random_state: Number,
    steps: [...]
  },
  output_files: [{
    name: String,
    path: String,
    rows: Number,
    columns: Number
  }],
  metadata: Mixed,
  transformation_log: [String],
  python_code: String,
  created_at: Date
}
```

## Testing

### Test Cases

1. **Clean Only**: Upload dataset → Select clean_only → Run → Verify cleaned output
2. **Split Only**: Upload dataset → Select split → Set test_size=0.3 → Run → Verify train/test files
3. **Clean + Split**: Upload dataset → Configure cleaning → Select clean_and_split → Run → Verify
4. **Cross-Validation**: Upload dataset → Select cross_validation → Set K=5 → Run → Verify 10 files (5 folds × 2)
5. **Clean + CV**: Upload dataset → Configure cleaning → Select clean_and_cv → Set K=3 → Run → Verify

### Validation Tests

- Test with dataset too small (< 10 rows)
- Test with invalid test_size (0.5)
- Test with invalid n_folds (15)
- Test with invalid operation_type

## Dependencies

### Frontend
- React 18+
- Axios
- CSS3

### Backend
- Node.js
- Express
- Mongoose
- Axios

### ML Service
- Python 3.8+
- FastAPI
- Pandas
- NumPy
- scikit-learn
- Pydantic

## Future Enhancements

- [ ] Stratified split for classification problems
- [ ] Time series split
- [ ] Group K-fold
- [ ] Automated feature selection
- [ ] Pipeline templates
- [ ] Batch processing
- [ ] Pipeline versioning
- [ ] A/B testing support

## Troubleshooting

### Issue: Pipeline fails with "Dataset too small"
**Solution**: Ensure dataset has at least 10 rows for splits, K×2 for K-fold CV

### Issue: Files not downloading
**Solution**: Check uploads directory permissions and file paths

### Issue: Python service timeout
**Solution**: Increase timeout for large datasets or complex pipelines

### Issue: Folds have unequal sizes
**Solution**: This is expected for datasets where rows aren't divisible by K

## License

MIT License - See main project README
