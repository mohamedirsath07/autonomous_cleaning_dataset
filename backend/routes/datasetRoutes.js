const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const archiver = require('archiver');
const Dataset = require('../models/Dataset');
const Profile = require('../models/Profile');
const PipelineRun = require('../models/PipelineRun');
const axios = require('axios');

// Configure Multer for file upload
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({ storage: storage });

// POST /api/datasets/upload - Upload dataset with auto-profiling (new endpoint for AutoKlean)
router.post('/upload', upload.single('file'), async (req, res) => {
    console.log("Received file upload request (AutoKlean)");
    try {
        if (!req.file) {
            console.log("No file in request");
            return res.status(400).json({ message: 'No file uploaded' });
        }
        console.log("File uploaded:", req.file);

        const newDataset = new Dataset({
            name: req.file.originalname,
            file_path: req.file.path,
        });

        const savedDataset = await newDataset.save();
        console.log("Dataset saved to DB:", savedDataset);

        // Auto-profile the dataset
        const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';
        const backendUrl = process.env.BACKEND_URL || `http://localhost:${process.env.PORT || 5000}`;
        const fileUrl = `${backendUrl}/uploads/${path.basename(savedDataset.file_path)}`;
        
        try {
            const profileResponse = await axios.post(`${pythonServiceUrl}/profile`, {
                file_path: fileUrl
            });

            // Save profile to database
            const newProfile = new Profile({
                dataset_id: savedDataset._id,
                columns: profileResponse.data.columns,
                rows: profileResponse.data.rows
            });
            await newProfile.save();

            // Return dataset with profile info
            res.status(201).json({
                _id: savedDataset._id,
                filename: savedDataset.name,
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
            // Return dataset without profile if profiling fails
            res.status(201).json({
                _id: savedDataset._id,
                filename: savedDataset.name,
                fileSize: req.file.size,
                profile: null
            });
        }
    } catch (error) {
        console.error("Error in upload route:", error);
        res.status(500).json({ message: 'Server error during upload' });
    }
});

// POST /api/datasets/:id/clean - Clean dataset with configuration options (updated for AutoKlean)
router.post('/:id/clean', async (req, res) => {
    console.log(`Clean request for ID: ${req.params.id}`);
    try {
        const dataset = await Dataset.findById(req.params.id);
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
        const backendUrl = process.env.BACKEND_URL || `http://localhost:${process.env.PORT || 5000}`;
        const fileUrl = `${backendUrl}/uploads/${path.basename(dataset.file_path)}`;
        
        console.log("Sending auto-clean request to Python service with config:", {
            splitRatio, kFolds, epochs, removeOutliers, imputeMissing, normalizeFeatures
        });

        const response = await axios.post(`${pythonServiceUrl}/auto-clean`, {
            file_path: fileUrl,
            split_ratio: splitRatio,
            k_folds: kFolds,
            epochs: epochs,
            remove_outliers: removeOutliers,
            impute_missing: imputeMissing,
            normalize_features: normalizeFeatures
        });

        console.log("Received response from Python service (auto-clean)");
        
        // Save cleaned file from base64 content
        let cleanedFilePath = null;
        let downloadPath = null;
        
        if (response.data.cleaned_csv_content) {
            const timestamp = Date.now();
            const cleanedFileName = `${timestamp}_cleaned.csv`;
            cleanedFilePath = path.join(__dirname, '..', 'uploads', cleanedFileName);
            
            const cleanedData = Buffer.from(response.data.cleaned_csv_content, 'base64');
            fs.writeFileSync(cleanedFilePath, cleanedData);
            downloadPath = `/uploads/${cleanedFileName}`;
            console.log("Saved cleaned file to:", cleanedFilePath);
        }
        
        // Save train/test files if provided
        let trainFilePath = null;
        let testFilePath = null;
        let splitZipPath = null;
        
        if (response.data.train_csv_content && response.data.test_csv_content) {
            const timestamp = Date.now();
            const splitDir = path.join(__dirname, '..', 'uploads', `${timestamp}_split`);
            fs.mkdirSync(splitDir, { recursive: true });
            
            trainFilePath = path.join(splitDir, 'train.csv');
            testFilePath = path.join(splitDir, 'test.csv');
            
            const trainData = Buffer.from(response.data.train_csv_content, 'base64');
            const testData = Buffer.from(response.data.test_csv_content, 'base64');
            
            fs.writeFileSync(trainFilePath, trainData);
            fs.writeFileSync(testFilePath, testData);
            
            console.log("Saved train/test files to:", splitDir);
            
            // Create ZIP file
            const zipFileName = `${timestamp}_train_test.zip`;
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
        
        // Save pipeline run to database
        const newRun = new PipelineRun({
            dataset_id: dataset._id,
            pipeline_config: { 
                splitRatio, kFolds, epochs, removeOutliers, imputeMissing, normalizeFeatures
            },
            cleaned_file_path: cleanedFilePath,
            transformation_log: response.data.transformation_log || [],
            python_code: response.data.python_code || ''
        });
        const savedRun = await newRun.save();
        
        res.json({
            ...response.data,
            run_id: savedRun._id,
            cleanedFilePath: downloadPath,
            splitZipPath: splitZipPath
        });
    } catch (error) {
        console.error("Error in clean route:", error.message);
        if (error.response) {
            console.error("Python service error data:", error.response.data);
        }
        res.status(500).json({ message: error.response?.data?.detail || 'Error cleaning dataset' });
    }
});

// POST /api/datasets - Upload dataset (original endpoint)
router.post('/', upload.single('file'), async (req, res) => {
    console.log("Received file upload request");
    try {
        if (!req.file) {
            console.log("No file in request");
            return res.status(400).json({ message: 'No file uploaded' });
        }
        console.log("File uploaded:", req.file);

        const newDataset = new Dataset({
            name: req.file.originalname,
            file_path: req.file.path,
        });

        const savedDataset = await newDataset.save();
        console.log("Dataset saved to DB:", savedDataset);
        res.status(201).json(savedDataset);
    } catch (error) {
        console.error("Error in upload route:", error);
        res.status(500).json({ message: 'Server error during upload' });
    }
});

// GET /api/datasets/:id/profile - Fetch profiling result
router.get('/:id/profile', async (req, res) => {
    console.log(`Profiling request for ID: ${req.params.id}`);
    try {
        const dataset = await Dataset.findById(req.params.id);
        if (!dataset) {
            console.log("Dataset not found in DB");
            return res.status(404).json({ message: 'Dataset not found' });
        }

        // Call Python ML service to generate profile
        // Assuming Python service is running on port 8000
        const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';
        
        // We need to send the absolute path or a path accessible to the python service
        // For local dev, absolute path works if both are on same machine
        const absolutePath = path.resolve(dataset.file_path);
        console.log(`Sending profile request to Python service for file: ${absolutePath}`);

        const response = await axios.post(`${pythonServiceUrl}/profile`, {
            file_path: absolutePath
        });
        console.log("Received response from Python service");

        // Save profile to database
        const newProfile = new Profile({
            dataset_id: dataset._id,
            columns: response.data.columns,
            rows: response.data.rows
        });
        await newProfile.save();

        res.json(response.data);
    } catch (error) {
        console.error("Error in profile route:", error.message);
        if (error.response) {
            console.error("Python service error data:", error.response.data);
        }
        res.status(500).json({ message: 'Error generating profile' });
    }
});

// POST /api/datasets/:id/prepare - Apply cleaning pipeline
router.post('/:id/prepare', async (req, res) => {
    console.log(`Prepare request for ID: ${req.params.id}`);
    try {
        const dataset = await Dataset.findById(req.params.id);
        if (!dataset) {
            return res.status(404).json({ message: 'Dataset not found' });
        }

        const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';
        const absolutePath = path.resolve(dataset.file_path);
        
        console.log("Sending prepare request to Python service with steps:", JSON.stringify(req.body.steps));

        const response = await axios.post(`${pythonServiceUrl}/prepare`, {
            file_path: absolutePath,
            steps: req.body.steps
        });

        console.log("Received response from Python service (prepare)");
        
        // Save pipeline run to database
        const newRun = new PipelineRun({
            dataset_id: dataset._id,
            pipeline_config: { steps: req.body.steps },
            cleaned_file_path: response.data.cleaned_file_path,
            transformation_log: response.data.transformation_log,
            python_code: response.data.python_code
        });
        const savedRun = await newRun.save();
        
        res.json({
            ...response.data,
            run_id: savedRun._id
        });
    } catch (error) {
        console.error("Error in prepare route:", error.message);
        if (error.response) {
            console.error("Python service error data:", error.response.data);
        }
        res.status(500).json({ message: 'Error preparing dataset' });
    }
});

// GET /api/runs - Get all pipeline runs
router.get('/runs', async (req, res) => {
    try {
        const runs = await PipelineRun.find()
            .populate('dataset_id')
            .sort({ created_at: -1 })
            .limit(50);
        res.json(runs);
    } catch (error) {
        console.error("Error fetching runs:", error);
        res.status(500).json({ message: 'Error fetching runs' });
    }
});

// GET /api/runs/:id - Get specific run details
router.get('/runs/:id', async (req, res) => {
    try {
        const run = await PipelineRun.findById(req.params.id).populate('dataset_id');
        if (!run) {
            return res.status(404).json({ message: 'Run not found' });
        }
        res.json(run);
    } catch (error) {
        console.error("Error fetching run:", error);
        res.status(500).json({ message: 'Error fetching run' });
    }
});

// GET /api/runs/:id/download - Download cleaned file
router.get('/runs/:id/download', async (req, res) => {
    try {
        const run = await PipelineRun.findById(req.params.id);
        if (!run) {
            return res.status(404).json({ message: 'Run not found' });
        }
        
        const filePath = path.resolve(run.cleaned_file_path);
        res.download(filePath);
    } catch (error) {
        console.error("Error downloading file:", error);
        res.status(500).json({ message: 'Error downloading file' });
    }
});

// POST /api/datasets/:id/run-pipeline - Run advanced pipeline operations
router.post('/:id/run-pipeline', async (req, res) => {
    console.log(`Run pipeline request for ID: ${req.params.id}`);
    try {
        const dataset = await Dataset.findById(req.params.id);
        if (!dataset) {
            return res.status(404).json({ message: 'Dataset not found' });
        }

        const { 
            operation_type, 
            test_size, 
            n_folds, 
            shuffle, 
            random_state, 
            cleaning_pipeline 
        } = req.body;

        // Validation
        const validOperations = ['clean_only', 'split', 'clean_and_split', 'cross_validation', 'clean_and_cv'];
        if (!operation_type || !validOperations.includes(operation_type)) {
            return res.status(400).json({ message: 'Invalid operation_type' });
        }

        if ((operation_type === 'split' || operation_type === 'clean_and_split') && 
            (test_size < 0.1 || test_size > 0.4)) {
            return res.status(400).json({ message: 'test_size must be between 0.1 and 0.4' });
        }

        if ((operation_type === 'cross_validation' || operation_type === 'clean_and_cv') && 
            (n_folds < 2 || n_folds > 10)) {
            return res.status(400).json({ message: 'n_folds must be between 2 and 10' });
        }

        const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';
        const absolutePath = path.resolve(dataset.file_path);
        
        console.log("Sending run-pipeline request to Python service");

        const response = await axios.post(`${pythonServiceUrl}/run-pipeline`, {
            file_path: absolutePath,
            operation_type,
            test_size: test_size || 0.2,
            n_folds: n_folds || 5,
            shuffle: shuffle !== undefined ? shuffle : true,
            random_state: random_state || 42,
            cleaning_pipeline: cleaning_pipeline || { steps: [] }
        });

        console.log("Received response from Python service (run-pipeline)");
        
        // Save pipeline run to database
        const newRun = new PipelineRun({
            dataset_id: dataset._id,
            pipeline_config: { 
                operation_type,
                test_size,
                n_folds,
                shuffle,
                random_state,
                steps: cleaning_pipeline?.steps || []
            },
            cleaned_file_path: response.data.output_files?.[0]?.path || '',
            output_files: response.data.output_files || [],
            metadata: response.data.metadata || {},
            transformation_log: response.data.transformation_log || [],
            python_code: response.data.python_code || ''
        });
        const savedRun = await newRun.save();
        
        res.json({
            ...response.data,
            run_id: savedRun._id
        });
    } catch (error) {
        console.error("Error in run-pipeline route:", error.message);
        if (error.response) {
            console.error("Python service error data:", error.response.data);
        }
        res.status(500).json({ message: error.response?.data?.detail || 'Error running pipeline' });
    }
});

module.exports = router;
