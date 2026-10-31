# 🧹 AutoKlean - Autonomous Data Preparation Engine

<div align="center">

![AutoKlean](https://img.shields.io/badge/AutoKlean-v2.0-brightgreen?style=for-the-badge)
![React](https://img.shields.io/badge/React-19.2-61DAFB?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)
![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=for-the-badge&logo=node.js)
![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47A248?style=for-the-badge&logo=mongodb)

**A full-stack web application for autonomous data cleaning, profiling, and ML-ready dataset preparation.**

[Quick Start](#-quick-start) • [Features](#-features) • [Architecture](#-architecture) • [API Reference](#-api-reference)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [API Reference](#-api-reference)
- [Configuration Options](#-configuration-options)

---

## 🎯 Overview

**AutoKlean** is an intelligent data preparation platform that automates the tedious process of cleaning and preparing raw datasets for machine learning. Simply upload your CSV or Excel file, and AutoKlean will:

1. **Automatically profile** your data (detect types, find missing values, identify outliers)
2. **Clean and transform** data using intelligent pipelines
3. **Generate train/test splits** or K-fold cross-validation sets
4. **Export ML-ready datasets** with full transformation logs

### Why AutoKlean?

| Traditional Approach | With AutoKlean |
|---------------------|----------------|
| Hours of manual data inspection | Instant automated profiling |
| Custom scripts for each dataset | One-click intelligent cleaning |
| Error-prone manual splitting | Configurable train/test & K-fold splits |
| No reproducibility | Generated Python code for every pipeline |

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| 📤 **Smart Upload** | Drag-and-drop CSV/Excel files with instant validation |
| 🔍 **Auto-Profiling** | Detects data types, missing values, outliers, and quality metrics |
| 🧹 **Intelligent Cleaning** | Handles missing data, normalizes values, encodes categories |
| ✂️ **Train/Test Split** | Configurable 0-100% split ratio with random seed control |
| 🔄 **K-Fold CV** | Generate 2-20 fold cross-validation sets |
| 📊 **Quality Metrics** | Completeness, uniqueness, and distribution analysis |
| 🐍 **Python Export** | Auto-generated reproducible Python code |
| 📦 **Batch Download** | Download all outputs as ZIP |

### Pipeline Operations

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE OPTIONS                         │
├─────────────────────────────────────────────────────────────┤
│  1. Clean Only        → cleaned.csv                         │
│  2. Split Only        → train.csv + test.csv                │
│  3. Clean + Split     → cleaned_train.csv + cleaned_test.csv│
│  4. K-Fold CV         → fold_1_train.csv ... fold_k_val.csv │
│  5. Clean + K-Fold    → cleaned K-fold files                │
└─────────────────────────────────────────────────────────────┘
```

### Cleaning Operations

- **Missing Value Imputation**: Mean, median, mode, or KNN-based
- **Outlier Removal**: IsolationForest-based detection
- **Feature Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **Encoding**: Label encoding, one-hot encoding
- **Type Coercion**: Automatic numeric/datetime/boolean detection
- **Schema Cleaning**: Column renaming to snake_case

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser)                          │
│                    React 19 + Vite + TailwindCSS                  │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTP (REST API)
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      BACKEND (Node.js/Express)                    │
│  • File Upload (Multer)     • Dataset Management                  │
│  • MongoDB Integration      • Pipeline Orchestration              │
│  • ZIP Generation           • Static File Serving                 │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTP (Internal API)
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      ML SERVICE (Python/FastAPI)                  │
│  • Data Profiling           • Auto-Cleaning Pipeline              │
│  • Train/Test Splitting     • K-Fold Cross-Validation             │
│  • Outlier Detection        • Feature Engineering                 │
│  • Python Code Generation   • Transformation Logging              │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         STORAGE                                   │
│  • MongoDB (metadata, profiles, pipeline runs)                    │
│  • File System (uploaded & processed datasets)                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | React 19, Vite 7, TailwindCSS 4 | Modern reactive UI |
| **Backend** | Node.js, Express 5, Mongoose 9 | REST API & file handling |
| **ML Service** | Python 3.12, FastAPI, scikit-learn | Data processing & ML ops |
| **Database** | MongoDB 7 | Document storage |
| **Styling** | TailwindCSS, Lucide Icons | UI components |

---

## 📁 Project Structure

```
Autonomous_Data_Cleaning/
├── frontend/                    # React Frontend Application
│   ├── src/
│   │   ├── App.jsx             # Main application component
│   │   ├── App.css             # Custom styles
│   │   └── main.jsx            # Entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── backend/                     # Node.js Backend API
│   ├── server.js               # Express server entry
│   ├── routes/
│   │   └── datasetRoutes.js    # API endpoints
│   ├── models/
│   │   ├── Dataset.js          # Dataset schema
│   │   ├── Profile.js          # Profile schema
│   │   └── PipelineRun.js      # Pipeline run schema
│   ├── uploads/                # Uploaded & processed files
│   └── package.json
│
├── ml-service/                  # Python ML Service
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   └── test_pipeline.py        # Test scripts
│
├── README.md                   # This file
├── QUICK_START.md              # Setup guide
├── PIPELINE_OPERATIONS.md      # Feature documentation
├── IMPLEMENTATION_SUMMARY.md   # Technical details
└── TESTING_GUIDE.md            # Testing checklist
```

---

## 🚀 Quick Start

> Deployment note: the frontend and Python service must be deployed as separate public services. The backend requires `PYTHON_SERVICE_URL` and `BACKEND_URL` to point to the live services, otherwise upload profiling and cleaning will fail in production.

### Prerequisites

- **Node.js** v18+ 
- **Python** 3.10+
- A running Python ML service on `localhost:8000` or a deployed equivalent

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Autonomous_Data_Cleaning.git
cd Autonomous_Data_Cleaning

# Install backend dependencies
cd backend
npm install

# Install frontend dependencies
cd ../frontend
npm install

# Install Python dependencies
cd ../ml-service
pip install -r requirements.txt
```

### Running the Application

Open **3 terminals** and run:

```bash
# Terminal 1: Backend (Port 5000)
cd backend
node server.js

# Terminal 2: ML Service (Port 8000)
cd ml-service
python main.py

# Terminal 3: Frontend (Port 5173)
cd frontend
npm run dev
```

### Access the Application

Open your browser and navigate to: **http://localhost:5173**

---

### Production environment variables

Configure these on the backend deployment:

```text
PYTHON_SERVICE_URL=https://your-ml-service.example.com
BACKEND_URL=https://your-backend.example.com
MAX_UPLOAD_SIZE=50mb
```

Deploy `ml-service` as a separate Python web service with:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The supported upload formats are `.csv`, `.xlsx`, and `.xls`. JSON is not currently supported.

## ⚙️ How It Works

### 1. Upload Dataset
```
User uploads CSV/Excel → Backend saves file → ML Service profiles data
                                            ↓
                              Returns: column types, missing %, stats
```

### 2. Configure Pipeline
```
User selects options:
  ├── Train/Test Split Ratio (0-100%)
  ├── K-Folds for Cross-Validation (0-20)
  ├── Remove Outliers (on/off)
  ├── Impute Missing Values (on/off)
  └── Normalize Features (on/off)
```

### 3. Execute Pipeline
```
Backend receives config → Calls ML Service /auto-clean endpoint
                                    ↓
ML Service performs:
  1. Schema cleaning (column names, type coercion)
  2. Semantic cleaning (missing tokens, standardization)
  3. Outlier removal (if enabled)
  4. Missing value imputation (if enabled)
  5. Feature normalization (if enabled)
  6. Train/Test split or K-Fold generation
  7. Save processed files
  8. Generate Python code
                                    ↓
Returns: file paths, transformation log, Python code
```

### 4. Download Results
```
User downloads:
  ├── cleaned.csv (cleaned dataset)
  ├── train.csv + test.csv (if split enabled)
  ├── fold_X_train.csv + fold_X_val.csv (if K-fold enabled)
  └── pipeline_code.py (reproducible Python script)
```

---

## 📡 API Reference

### Backend Endpoints (Port 5000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/datasets/upload` | Upload file & auto-profile |
| `GET` | `/api/datasets/:id/profile` | Get dataset profile |
| `POST` | `/api/datasets/:id/clean` | Run cleaning pipeline |
| `GET` | `/uploads/:filename` | Download processed files |

### ML Service Endpoints (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/profile` | Profile a dataset |
| `POST` | `/auto-clean` | Run full cleaning pipeline |
| `GET` | `/docs` | Swagger API documentation |

### Example: Run Pipeline

```bash
curl -X POST http://localhost:5000/api/datasets/DATASET_ID/clean \
  -H "Content-Type: application/json" \
  -d '{
    "splitRatio": 80,
    "kFolds": 5,
    "removeOutliers": true,
    "imputeMissing": true,
    "normalizeFeatures": true
  }'
```

---

## 🎛️ Configuration Options

| Option | Type | Range | Default | Description |
|--------|------|-------|---------|-------------|
| `splitRatio` | number | 0-100 | 0 | Train set percentage (0 = no split) |
| `kFolds` | number | 0-20 | 0 | Number of CV folds (0 = disabled) |
| `epochs` | number | 0+ | 0 | Reserved for future use |
| `removeOutliers` | boolean | - | false | Enable outlier removal |
| `imputeMissing` | boolean | - | false | Enable missing value imputation |
| `normalizeFeatures` | boolean | - | false | Enable feature scaling |

---

## 🖼️ UI Features

### Main Interface
- **Dark theme** with neon green (#ccff00) accents
- **Drag-and-drop** file upload zone
- **Real-time** processing logs in terminal-style panel
- **Bento grid** layout for data visualization

### Pipeline Config Panel
- **Number input + slider** for precise train/test split control
- **Increment/decrement buttons** with number input for K-folds
- **Toggle switches** for cleaning options (all off by default)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">

**Built with ❤️ for the ML community**

</div>
