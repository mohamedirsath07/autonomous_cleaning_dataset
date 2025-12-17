# Quick Start Guide - Advanced Pipeline Operations

## Prerequisites

1. **MongoDB** running on `localhost:27017`
2. **Node.js** v14+ installed
3. **Python** 3.8+ with pip installed

## Installation & Setup

### 1. Install Backend Dependencies

```powershell
cd backend
npm install
```

### 2. Install Frontend Dependencies

```powershell
cd frontend
npm install
```

### 3. Install Python ML Service Dependencies

```powershell
cd ml-service
pip install -r requirements.txt
```

## Running the Application

### Terminal 1: Start Backend Server

```powershell
cd backend
node server.js
```

Should see: `Server running on port 5000` and `MongoDB connected`

### Terminal 2: Start Python ML Service

```powershell
cd ml-service
python main.py
```

Should see: `Uvicorn running on http://0.0.0.0:8000`

### Terminal 3: Start Frontend Development Server

```powershell
cd frontend
npm run dev
```

Should see: `Local: http://localhost:5173/`

## Using the Pipeline Operations

### Step 1: Upload Dataset
1. Open browser to `http://localhost:5173`
2. Click or drag-and-drop CSV/XLSX file
3. Click "Start Profiling 🚀"

### Step 2: Configure Cleaning (Optional)
- Review auto-suggested cleaning steps
- Modify actions for each column:
  - Impute (mean, median, mode)
  - Scale (standard, minmax)
  - Encode (label, onehot)
  - Drop column

### Step 3: Choose Pipeline Operation

#### Option 1: Clean Only
- Select "🧹 Clean Dataset Only"
- Click "🚀 Run Data Pipeline"
- Download `cleaned.csv`

#### Option 2: Split Only
- Select "✂️ Split Dataset (Train/Test)"
- Set test size (slider: 10-40%)
- Set random seed (default: 42)
- Click "🚀 Run Data Pipeline"
- Download `train.csv` and `test.csv`

#### Option 3: Clean + Split
- Select "🧹✂️ Clean + Train/Test Split"
- Configure cleaning steps above
- Set test size and random seed
- Click "🚀 Run Data Pipeline"
- Download cleaned train/test files

#### Option 4: Cross-Validation
- Select "🔄 Cross-Validation Split"
- Set number of folds K (2-10)
- Check/uncheck shuffle
- Set random seed
- Click "🚀 Run Data Pipeline"
- Download K pairs of train/val files

#### Option 5: Clean + CV
- Select "🧹🔄 Clean + Cross-Validation"
- Configure cleaning steps
- Set K folds and shuffle
- Click "🚀 Run Data Pipeline"
- Download cleaned K-fold files

### Step 4: View Results
- **Output Files**: Download each file individually
- **Metadata**: View row/column counts, split ratios
- **Transformation Log**: See what operations were applied
- **Python Code**: Copy generated code for reproduction

## Example Workflow

### Scenario: Prepare data for ML model training

```
1. Upload: spotify_2015_2025_85k.csv
2. Profile: System detects missing values, outliers
3. Clean: 
   - Impute 'streams' with median
   - Scale 'duration_ms' with standard scaler
   - Encode 'artist' with label encoder
4. Split: Choose "Clean + Train/Test Split"
   - Set test_size = 0.2 (80/20 split)
   - Random seed = 42
5. Run: Click "Run Data Pipeline"
6. Download:
   - train.csv (68,000 rows)
   - test.csv (17,000 rows)
7. Copy generated Python code for your ML pipeline
```

## Troubleshooting

### Backend won't start
```powershell
# Check MongoDB is running
mongosh

# Check port 5000 is not in use
netstat -ano | findstr :5000
```

### Python service won't start
```powershell
# Verify Python version
python --version  # Should be 3.8+

# Check port 8000 is not in use
netstat -ano | findstr :8000

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Frontend won't start
```powershell
# Clear node_modules and reinstall
Remove-Item -Recurse -Force node_modules
npm install

# Try different port
npm run dev -- --port 3000
```

### "Dataset not found" error
- Ensure file uploaded successfully
- Check `backend/uploads/` folder exists
- Verify MongoDB connection

### "Python service error"
- Check Python service is running on port 8000
- View Python service terminal for error details
- Verify file paths are correct

## File Locations

### Uploads
```
backend/uploads/
├── [timestamp]-original.csv
├── [timestamp]_clean_only/
│   └── cleaned.csv
├── [timestamp]_split/
│   ├── train.csv
│   └── test.csv
└── [timestamp]_clean_and_cv/
    ├── fold_1_train.csv
    ├── fold_1_val.csv
    └── ...
```

### Database
- **Database**: `autonomous_data_engine`
- **Collections**: `datasets`, `profiles`, `pipelineruns`

## API Endpoints Quick Reference

### Backend (Port 5000)
- `POST /api/datasets` - Upload dataset
- `GET /api/datasets/:id/profile` - Get profile
- `POST /api/datasets/:id/prepare` - Clean dataset (legacy)
- `POST /api/datasets/:id/run-pipeline` - **Run advanced pipeline** ⭐
- `GET /api/datasets/runs` - Get pipeline history
- `GET /api/datasets/runs/:id/download` - Download result

### Python Service (Port 8000)
- `POST /profile` - Profile dataset
- `POST /prepare` - Prepare dataset (legacy)
- `POST /run-pipeline` - **Execute pipeline operations** ⭐

## Testing the New Features

### Test 1: Basic Split
```powershell
# In frontend, upload a CSV with 1000 rows
# Select "Split Dataset (Train/Test)"
# Set test_size = 0.2
# Expected: train.csv (800 rows), test.csv (200 rows)
```

### Test 2: 5-Fold CV
```powershell
# Upload a CSV with 1000 rows
# Select "Cross-Validation Split"
# Set K = 5
# Expected: 10 files (fold_1 through fold_5, each with train and val)
```

### Test 3: Clean + Split
```powershell
# Upload CSV with missing values
# Configure imputation for columns
# Select "Clean + Train/Test Split"
# Set test_size = 0.3
# Expected: cleaned train.csv (70%), test.csv (30%)
```

## Next Steps

1. **Explore History**: View past pipeline runs in the history sidebar
2. **Generate Code**: Copy Python code to reproduce pipelines locally
3. **Experiment**: Try different split ratios and fold counts
4. **Integrate**: Use downloaded files in your ML workflows
5. **Automate**: Use the API endpoints for batch processing

## Support

- 📚 Full Documentation: `PIPELINE_OPERATIONS.md`
- 🐛 Issues: Check terminal logs for errors
- 💡 Tips: Use random seed for reproducible splits

## Performance Tips

- Large files (>100MB): May take 30-60 seconds
- Cross-validation: K=5 recommended for most cases
- Test size: 20-30% is standard for ML
- Random seed: 42 is conventional (but any integer works)

---

**Ready to transform your data!** 🚀
