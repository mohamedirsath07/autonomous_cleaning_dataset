# Testing Guide - Advanced Pipeline Operations

## Pre-Testing Setup

### 1. Start All Services

```powershell
# Terminal 1: Backend
cd backend
node server.js
# Wait for: "Server running on port 5000" and "MongoDB connected"

# Terminal 2: Python ML Service  
cd ml-service
python main.py
# Wait for: "Uvicorn running on http://0.0.0.0:8000"

# Terminal 3: Frontend
cd frontend
npm run dev
# Wait for: "Local: http://localhost:5173/"
```

### 2. Verify MongoDB is Running
```powershell
mongosh
# Should connect successfully
```

### 3. Prepare Test Data
- Ensure at least one CSV file exists in `backend/uploads/`
- Recommended: Use a dataset with 1000+ rows for proper testing

## Test Suite

### Test 1: Basic UI Navigation ✅

**Steps:**
1. Open http://localhost:5173
2. Upload a CSV file
3. Click "Start Profiling 🚀"
4. Wait for profile to load

**Expected Results:**
- ✅ File upload successful
- ✅ Profile displays with columns
- ✅ Data quality metrics shown
- ✅ Auto-suggested cleaning steps applied

**Validation:**
- [ ] No console errors
- [ ] All columns displayed
- [ ] Statistics (rows, columns, missing %, unique) shown

---

### Test 2: Clean Only Operation ✅

**Steps:**
1. After profiling, scroll to "Advanced Dataset Operations"
2. Select "🧹 Clean Dataset Only"
3. Verify cleaning steps in profile builder (optional: modify)
4. Click "🚀 Run Data Pipeline"
5. Wait for completion

**Expected Results:**
- ✅ Pipeline completes successfully
- ✅ 1 output file: `cleaned.csv`
- ✅ Download button appears
- ✅ Transformation log shows cleaning steps
- ✅ Metadata shows rows before/after

**Validation:**
- [ ] `cleaned.csv` downloads successfully
- [ ] File opens in Excel/text editor
- [ ] Row count matches metadata
- [ ] No errors in console

---

### Test 3: Split Operation (No Cleaning) ✅

**Steps:**
1. Upload new file or use existing
2. Profile the dataset
3. Select "✂️ Split Dataset (Train/Test)"
4. Set test size to 30% using slider
5. Set random seed to 42
6. Click "🚀 Run Data Pipeline"

**Expected Results:**
- ✅ Pipeline completes successfully
- ✅ 2 output files: `train.csv`, `test.csv`
- ✅ Train set has ~70% of rows
- ✅ Test set has ~30% of rows
- ✅ Total rows = original rows

**Validation:**
- [ ] Both files download successfully
- [ ] Row distribution correct (70/30)
- [ ] No data overlap between train and test
- [ ] Columns match original dataset

**Calculations:**
```
If original = 1000 rows:
- Train should have ~700 rows
- Test should have ~300 rows
- Total = 1000 rows
```

---

### Test 4: Clean + Split Operation ✅

**Steps:**
1. Upload dataset with missing values
2. Profile and configure cleaning:
   - Impute numeric column with median
   - Scale a column with standard scaler
3. Select "🧹✂️ Clean + Train/Test Split"
4. Set test size to 20%
5. Click "🚀 Run Data Pipeline"

**Expected Results:**
- ✅ Pipeline completes successfully
- ✅ 2 output files: `train.csv`, `test.csv`
- ✅ Both files are cleaned (no missing values in imputed columns)
- ✅ Transformation log shows cleaning + split
- ✅ Metadata shows rows_after_cleaning

**Validation:**
- [ ] Files are cleaned (check imputed column)
- [ ] Train: 80%, Test: 20%
- [ ] Scaled column has mean ~0, std ~1 (in train set)
- [ ] No missing values in imputed columns

---

### Test 5: Cross-Validation (5 Folds) ✅

**Steps:**
1. Upload dataset (minimum 50 rows recommended)
2. Profile dataset
3. Select "🔄 Cross-Validation Split"
4. Set number of folds: 5
5. Check "Shuffle data before splitting"
6. Set random seed: 123
7. Click "🚀 Run Data Pipeline"

**Expected Results:**
- ✅ Pipeline completes successfully
- ✅ 10 output files:
  - `fold_1_train.csv`, `fold_1_val.csv`
  - `fold_2_train.csv`, `fold_2_val.csv`
  - `fold_3_train.csv`, `fold_3_val.csv`
  - `fold_4_train.csv`, `fold_4_val.csv`
  - `fold_5_train.csv`, `fold_5_val.csv`
- ✅ Each validation set has ~20% of data
- ✅ Each train set has ~80% of data

**Validation:**
- [ ] All 10 files present
- [ ] Total rows across all folds = original rows × 5
- [ ] Each fold non-overlapping
- [ ] File sizes roughly equal

**Calculations:**
```
If original = 1000 rows, K = 5:
- Each validation fold: ~200 rows
- Each training fold: ~800 rows
- Total downloads: 10 files
```

---

### Test 6: Clean + Cross-Validation ✅

**Steps:**
1. Upload dataset with data quality issues
2. Configure cleaning pipeline:
   - Impute missing values
   - Encode categorical column
3. Select "🧹🔄 Clean + Cross-Validation"
4. Set folds: 3
5. Enable shuffle
6. Click "🚀 Run Data Pipeline"

**Expected Results:**
- ✅ Pipeline completes successfully
- ✅ 6 output files (3 folds × 2)
- ✅ All files are cleaned
- ✅ Transformation log shows cleaning + CV split
- ✅ Metadata shows cleaning and fold stats

**Validation:**
- [ ] All files cleaned (no missing values)
- [ ] 3 complete folds
- [ ] Each validation set has ~33% of cleaned data
- [ ] Encoded column present in all files

---

### Test 7: Edge Cases ⚠️

#### Test 7.1: Small Dataset (< 10 rows)
**Steps:**
1. Upload CSV with 5 rows
2. Try "Split Dataset" operation

**Expected:**
- ❌ Error message: "Dataset too small for splitting"

#### Test 7.2: Invalid Test Size
**Steps:**
1. Manually set test_size = 0.5 (via browser console)
2. Try split operation

**Expected:**
- ❌ Backend validation error

#### Test 7.3: Invalid Fold Count
**Steps:**
1. Set folds to 15
2. Try CV operation

**Expected:**
- ❌ Validation error: "n_folds must be between 2 and 10"

#### Test 7.4: Dataset Too Small for K-Fold
**Steps:**
1. Upload CSV with 8 rows
2. Try K=5 cross-validation

**Expected:**
- ❌ Error: "Dataset too small for 5-fold CV"

---

### Test 8: Python Code Generation ✅

**Steps:**
1. Run any pipeline operation
2. After completion, click "Show Python Code"
3. Copy the code
4. Paste in a Python file and verify syntax

**Expected Results:**
- ✅ Valid Python code generated
- ✅ Imports included
- ✅ Code matches pipeline operations
- ✅ Can be executed independently

**Validation:**
- [ ] Code has proper imports
- [ ] File paths are correct
- [ ] Logic matches selected operation
- [ ] Code is well-formatted

---

### Test 9: History Tracking ✅

**Steps:**
1. Run 3 different pipeline operations
2. Click "📜 History (3)" button
3. View history sidebar

**Expected Results:**
- ✅ All 3 runs shown
- ✅ Timestamps displayed
- ✅ Dataset names shown
- ✅ Download buttons work

**Validation:**
- [ ] History count correct
- [ ] Can download from history
- [ ] Most recent run at top
- [ ] All metadata preserved

---

### Test 10: Multiple Sequential Operations ✅

**Steps:**
1. Upload dataset
2. Run "Clean Only"
3. Without refreshing, run "Split" on same dataset
4. Run "Clean + CV"

**Expected Results:**
- ✅ All operations complete successfully
- ✅ Each creates separate output folder
- ✅ No interference between operations
- ✅ All stored in database

**Validation:**
- [ ] 3 separate output directories
- [ ] All files accessible
- [ ] Database has 3 pipeline runs
- [ ] No file conflicts

---

## Performance Testing

### Test 11: Large Dataset (100K+ rows)

**Steps:**
1. Upload large CSV (100K rows)
2. Run "Clean + Split" operation
3. Monitor time and memory

**Expected:**
- ⏱️ Completion time: 10-30 seconds
- 💾 No memory errors
- ✅ Files created successfully

**Metrics to Track:**
- [ ] Upload time
- [ ] Profile time
- [ ] Pipeline execution time
- [ ] Download time

---

## Browser Compatibility

Test in multiple browsers:
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari (if on Mac)

---

## API Testing (Optional)

### Test with curl or Postman

```powershell
# Test run-pipeline endpoint directly
curl -X POST http://localhost:8000/run-pipeline `
  -H "Content-Type: application/json" `
  -d '{
    "file_path": "backend/uploads/your-file.csv",
    "operation_type": "split",
    "test_size": 0.2,
    "shuffle": true,
    "random_state": 42,
    "cleaning_pipeline": {"steps": []}
  }'
```

---

## Automated Test Script

Use the provided test script:

```powershell
cd ml-service
python test_pipeline.py
```

---

## Bug Report Template

If issues found:

```
**Issue:** [Brief description]
**Operation:** [clean_only | split | etc.]
**Steps to Reproduce:**
1. 
2. 
3. 

**Expected:** 
**Actual:** 
**Console Errors:** 
**Dataset Size:** [rows × columns]
```

---

## Success Criteria

### Critical (Must Pass) ✅
- [ ] All 5 operations complete successfully
- [ ] Files download correctly
- [ ] No data loss
- [ ] Correct split ratios
- [ ] Proper validation

### Important (Should Pass) ✅
- [ ] UI responsive and intuitive
- [ ] Error messages clear
- [ ] Python code generates correctly
- [ ] History tracking works
- [ ] Metadata accurate

### Nice to Have ✅
- [ ] Fast performance
- [ ] Smooth animations
- [ ] No console warnings
- [ ] Works in all browsers

---

## Final Checklist

Before marking as "Ready for Production":

- [ ] All core tests passed
- [ ] No critical bugs found
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] Edge cases handled
- [ ] Performance acceptable
- [ ] Security considerations addressed
- [ ] Database schema updated
- [ ] API endpoints documented

---

## Test Results Log

| Test | Status | Date | Notes |
|------|--------|------|-------|
| Test 1: UI Navigation | ⏳ | | |
| Test 2: Clean Only | ⏳ | | |
| Test 3: Split | ⏳ | | |
| Test 4: Clean + Split | ⏳ | | |
| Test 5: Cross-Validation | ⏳ | | |
| Test 6: Clean + CV | ⏳ | | |
| Test 7: Edge Cases | ⏳ | | |
| Test 8: Python Code | ⏳ | | |
| Test 9: History | ⏳ | | |
| Test 10: Sequential Ops | ⏳ | | |

Legend: ⏳ Pending | ✅ Passed | ❌ Failed | ⚠️ Issues Found

---

**Good luck with testing!** 🚀

Report any issues found and we'll fix them promptly.
