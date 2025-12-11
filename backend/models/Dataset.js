const mongoose = require('mongoose');

const datasetSchema = new mongoose.Schema({
    name: { type: String, required: true },
    owner: { type: String, default: 'anonymous' }, // Placeholder for user ID
    file_path: { type: String, required: true },
    uploaded_at: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Dataset', datasetSchema);
