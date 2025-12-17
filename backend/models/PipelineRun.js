const mongoose = require('mongoose');

const pipelineRunSchema = new mongoose.Schema({
    dataset_id: { type: mongoose.Schema.Types.ObjectId, ref: 'Dataset', required: true },
    pipeline_config: {
        operation_type: { type: String },
        test_size: { type: Number },
        n_folds: { type: Number },
        shuffle: { type: Boolean },
        random_state: { type: Number },
        steps: [{
            column: String,
            action: String,
            method: String,
            params: mongoose.Schema.Types.Mixed
        }]
    },
    cleaned_file_path: { type: String },
    output_files: [{
        name: String,
        path: String,
        rows: Number,
        columns: Number
    }],
    metadata: { type: mongoose.Schema.Types.Mixed },
    transformation_log: [String],
    python_code: { type: String },
    created_at: { type: Date, default: Date.now }
});

module.exports = mongoose.model('PipelineRun', pipelineRunSchema);
