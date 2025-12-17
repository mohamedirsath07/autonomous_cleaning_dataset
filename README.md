# Autonomous Data Preparation Engine

## Project Goal
Build a full-stack web application that allows users to upload raw datasets, automatically analyze and clean them, detect data quality issues, engineer features, and output a fully ML-ready dataset with advanced pipeline operations including train/test splits and cross-validation.

## ✨ Key Features

### Core Features
- 📤 **Upload Datasets**: Support for CSV and Excel files
- 🔍 **Automatic Profiling**: Intelligent data type detection, quality metrics, and issue identification
- 🧹 **Smart Cleaning**: Automated cleaning with customizable pipelines
- 📊 **ML-Ready Output**: Production-ready datasets with transformation logs

### 🚀 Advanced Pipeline Operations (NEW)
1. **Clean Dataset Only** - Apply cleaning pipeline without splitting
2. **Train/Test Split** - Split raw data into training and test sets (configurable 10-40%)
3. **Clean + Train/Test Split** - Clean first, then split for ML workflows
4. **Cross-Validation Split** - Create K-fold splits (2-10 folds) for robust evaluation
5. **Clean + Cross-Validation** - Combined cleaning and K-fold splitting

### Additional Features
- 🐍 **Python Code Generation**: Export reproducible Python code for each pipeline
- 📜 **Pipeline History**: Track and manage all data processing runs
- 📥 **Batch Downloads**: Download all output files (train, test, fold splits)
- 🎯 **Metadata Tracking**: Detailed statistics and transformation logs

## Tech Stack
- **Frontend**: React 18 (Vite)
- **Backend**: Node.js + Express + MongoDB
- **ML Service**: Python 3.8+ (FastAPI + scikit-learn)
- **Database**: MongoDB

## Project Structure
```
.
├── frontend/          # React application with advanced UI
├── backend/           # Node.js API with pipeline endpoints
├── ml-service/        # Python FastAPI service with ML operations
├── QUICK_START.md     # Quick setup and usage guide
├── PIPELINE_OPERATIONS.md  # Detailed feature documentation
├── TESTING_GUIDE.md   # Comprehensive testing checklist
└── IMPLEMENTATION_SUMMARY.md  # Technical implementation details
```
