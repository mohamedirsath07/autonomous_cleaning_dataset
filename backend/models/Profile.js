const mongoose = require('mongoose');

const profileSchema = new mongoose.Schema({
    dataset_id: { type: mongoose.Schema.Types.ObjectId, ref: 'Dataset', required: true },
    columns: [{
        name: String,
        inferred_type: String,
        dtype: String,
        missing_pct: Number,
        unique: Number,
        issues: [String],
        suggested: [String],
        sample_values: [mongoose.Schema.Types.Mixed]
    }],
    rows: { type: Number, required: true },
    profiled_at: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Profile', profileSchema);
