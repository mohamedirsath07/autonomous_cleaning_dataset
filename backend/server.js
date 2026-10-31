const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const axios = require('axios');
const archiver = require('archiver');

// Load .env from the backend directory
dotenv.config({ path: path.join(__dirname, '.env') });

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
    console.log('Created uploads directory');
}

const app = express();
const PORT = process.env.PORT || 5000;
const MAX_UPLOAD_SIZE_BYTES = (() => {
    const raw = process.env.MAX_UPLOAD_SIZE || '50mb';
    const match = raw.toLowerCase().match(/^([0-9]+)(mb|kb|b)?$/);
    if (!match) return 50 * 1024 * 1024;
    const value = Number.parseInt(match[1], 10);
    const unit = match[2] || 'b';
    if (unit === 'kb') return value * 1024;
    if (unit === 'mb') return value * 1024 * 1024;
    return value;
})();
const ALLOWED_EXTENSIONS = new Set(['.csv', '.xlsx', '.xls']);

// Middleware
app.use(cors({
    origin: true,
    credentials: true
}));
app.use(express.json());
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Configure Multer for file upload
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadsDir);
    },
    filename: (req, file, cb) => {
        const extension = path.extname(file.originalname).toLowerCase();
        const safeBaseName = path.basename(file.originalname, extension)
            .replace(/[^a-zA-Z0-9_-]/g, '_')
            .slice(0, 80) || 'dataset';
        cb(null, `${Date.now()}-${safeBaseName}${extension}`);
    }
});
const upload = multer({
    storage,
    limits: { fileSize: MAX_UPLOAD_SIZE_BYTES },
    fileFilter: (req, file, cb) => {
        const extension = path.extname(file.originalname).toLowerCase();
        if (!ALLOWED_EXTENSIONS.has(extension)) {
            return cb(new Error('Unsupported file type. Upload a CSV or Excel file.'));
        }
        cb(null, true);
    }
});

// In-memory storage for datasets (replaces MongoDB)
const datasets = new Map();
let datasetCounter = 0;

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// POST /api/datasets/upload - Upload dataset with auto-profiling
app.post('/api/datasets/upload', upload.single('file'), async (req, res) => {
    console.log("Received file upload request");
    try {
        if (!req.file) {
            console.log("No file in request");
            return res.status(400).json({ message: 'No file uploaded' });
        }
        console.log("File uploaded:", req.file);

        // Create dataset entry in memory
        datasetCounter++;
        const datasetId = `dataset_${datasetCounter}`;
        const dataset = {
            _id: datasetId,
            name: req.file.originalname,
            file_path: req.file.path,
            uploaded_at: new Date().toISOString()
        };
        datasets.set(datasetId, dataset);
        console.log("Dataset saved to memory:", dataset);

        // Auto-profile the dataset
        const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';
        const backendUrl = process.env.BACKEND_URL || `http://localhost:${PORT}`;
        const fileUrl = `${backendUrl}/uploads/${path.basename(dataset.file_path)}`;

        try {
            // Longer timeout for ML service cold start on free tier
            const profileResponse = await axios.post(`${pythonServiceUrl}/profile`, {
                file_path: fileUrl
            }, {
                timeout: 60000 // 60 seconds for cold start + profiling
            });

            // Return dataset with profile info
            res.status(201).json({
                _id: dataset._id,
                filename: dataset.name,
                fileSize: req.file.size,
                profile: {
                    shape: [profileResponse.data.rows, profileResponse.data.columns.length],
                    missing_values: profileResponse.data.columns.reduce((acc, col) => {
                        acc[col.name] = Math.round(col.missing_pct / 100 * profileResponse.data.rows);
                        return acc;
                    }, {}),
                    dtypes: profileResponse.data.columns.reduce((acc, col) => {
                        acc[col.name] = col.inferred_type;
                        return acc;
                    }, {})
                }
            });
        } catch (profileError) {
            console.error("Error profiling dataset:", profileError.message);
            return res.status(502).json({
                message: 'Dataset uploaded, but the profiling service is unavailable.',
                details: profileError.response?.data?.detail || profileError.code || 'Unknown profiling error',
                dataset: {
                    _id: dataset._id,
                    filename: dataset.name,
                    fileSize: req.file.size
                }
            });
        }
    } catch (error) {
        console.error("Error in upload route:", error);
        res.status(500).json({ message: 'Server error during upload' });
    }
});

// POST /api/datasets/:id/clean - Clean dataset with configuration options
app.post('/api/datasets/:id/clean', async (req, res) => {
    console.log(`Clean request for ID: ${req.params.id}`);
    try {
        const dataset = datasets.get(req.params.id);
        if (!dataset) {
            return res.status(404).json({ message: 'Dataset not found' });
        }

        const {
            splitRatio = 0.8,
            kFolds = 5,
            epochs = 100,
            removeOutliers = true,
            imputeMissing = true,
            normalizeFeatures = true
        } = req.body;

        const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';
        const backendUrl = process.env.BACKEND_URL || `http://localhost:${PORT}`;
        const fileUrl = `${backendUrl}/uploads/${path.basename(dataset.file_path)}`;

        console.log("Sending auto-clean request to Python service with config:", {
            splitRatio, kFolds, epochs, removeOutliers, imputeMissing, normalizeFeatures
        });

        // Use longer timeout for ML service (cold start can take 30-60 seconds on free tier)
        const response = await axios.post(`${pythonServiceUrl}/auto-clean`, {
            file_path: fileUrl,
            split_ratio: splitRatio,
            k_folds: kFolds,
            epochs: epochs,
            remove_outliers: removeOutliers,
            impute_missing: imputeMissing,
            normalize_features: normalizeFeatures
        }, {
            timeout: 120000, // 2 minutes timeout for cold start + processing
            headers: {
                'Content-Type': 'application/json'
            }
        });

        console.log("Received response from Python service (auto-clean)");

        // Save cleaned file from base64 content
        let cleanedFilePath = null;
        let downloadPath = null;

        if (response.data.cleaned_csv_content) {
            try {
                const timestamp = Date.now();
                const cleanedFileName = `autoklean_${timestamp}.csv`;

                // Ensure uploads directory exists
                if (!fs.existsSync(uploadsDir)) {
                    fs.mkdirSync(uploadsDir, { recursive: true });
                }

                cleanedFilePath = path.join(uploadsDir, cleanedFileName);

                const cleanedData = Buffer.from(response.data.cleaned_csv_content, 'base64');
                fs.writeFileSync(cleanedFilePath, cleanedData);
                downloadPath = `/uploads/${cleanedFileName}`;
                console.log("Saved cleaned file to:", cleanedFilePath);
            } catch (saveError) {
                console.error("Error saving cleaned file:", saveError);
            }
        }

        // Save train/test files if provided
        let splitZipPath = null;

        if (response.data.train_csv_content && response.data.test_csv_content) {
            const timestamp = Date.now();
            const splitDir = path.join(uploadsDir, `${timestamp}_split`);
            fs.mkdirSync(splitDir, { recursive: true });

            const trainFilePath = path.join(splitDir, 'train.csv');
            const testFilePath = path.join(splitDir, 'test.csv');

            const trainData = Buffer.from(response.data.train_csv_content, 'base64');
            const testData = Buffer.from(response.data.test_csv_content, 'base64');

            fs.writeFileSync(trainFilePath, trainData);
            fs.writeFileSync(testFilePath, testData);

            console.log("Saved train/test files to:", splitDir);

            // Create ZIP file
            const zipFileName = `autoklean_train_test_${timestamp}.zip`;
            const zipFilePath = path.join(splitDir, zipFileName);

            try {
                await new Promise((resolve, reject) => {
                    const output = fs.createWriteStream(zipFilePath);
                    const archive = archiver('zip', { zlib: { level: 9 } });

                    output.on('close', () => {
                        console.log("ZIP created successfully");
                        resolve();
                    });
                    archive.on('error', reject);

                    archive.pipe(output);
                    archive.file(trainFilePath, { name: 'train.csv' });
                    archive.file(testFilePath, { name: 'test.csv' });
                    archive.finalize();
                });

                splitZipPath = `/uploads/${timestamp}_split/${zipFileName}`;
            } catch (zipError) {
                console.error("Error creating ZIP:", zipError);
            }
        }

        res.json({
            ...response.data,
            cleanedFilePath: downloadPath,
            splitZipPath: splitZipPath
        });
    } catch (error) {
        console.error("Error in clean route:", error.message);
        if (error.response) {
            console.error("Python service error data:", error.response.data);
        }
        const upstreamMessage = error.response?.data?.detail || error.response?.data?.message;
        const status = error.code === 'ECONNABORTED' ? 504 : 502;
        res.status(status).json({
            message: upstreamMessage || 'The cleaning service is unavailable.',
            code: error.code || 'ML_SERVICE_ERROR'
        });
    }
});

// GET /api/datasets/:id/profile - Fetch profiling result
app.get('/api/datasets/:id/profile', async (req, res) => {
    console.log(`Profiling request for ID: ${req.params.id}`);
    try {
        const dataset = datasets.get(req.params.id);
        if (!dataset) {
            console.log("Dataset not found");
            return res.status(404).json({ message: 'Dataset not found' });
        }

        const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';
        const absolutePath = path.resolve(dataset.file_path);
        console.log(`Sending profile request to Python service for file: ${absolutePath}`);

        const response = await axios.post(`${pythonServiceUrl}/profile`, {
            file_path: absolutePath
        });
        console.log("Received response from Python service");

        res.json(response.data);
    } catch (error) {
        console.error("Error in profile route:", error.message);
        res.status(500).json({ message: 'Error generating profile' });
    }
});

app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            const sizeInMb = (MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)).toFixed(0);
            return res.status(413).json({ message: `File is too large. Maximum size is ${sizeInMb}MB.` });
        }
        return res.status(400).json({ message: `Upload failed: ${error.message}` });
    }
    if (error?.message?.startsWith('Unsupported file type')) {
        return res.status(415).json({ message: error.message });
    }
    next(error);
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log('MongoDB: DISABLED (using in-memory storage)');
});
