const mongoose = require('mongoose');

const pipelineRunSchema = new mongoose.Schema({
    dataset_id: { type: mongoose.Schema.Types.ObjectId, ref: 'Dataset', required: true },
    pipeline_config: {
        steps: [{
            column: String,
            action: String,
            method: String,
            params: mongoose.Schema.Types.Mixed
        }]
    },
    cleaned_file_path: { type: String, required: true },
    transformation_log: [String],
    python_code: { type: String },
    created_at: { type: Date, default: Date.now }
});

module.exports = mongoose.model('PipelineRun', pipelineRunSchema);
