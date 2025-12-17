const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
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

// POST /api/datasets - Upload dataset
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
