import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [dataset, setDataset] = useState(null);
  const [profile, setProfile] = useState(null);
  const [error, setError] = useState(null);
  
  // Pipeline State
  const [pipelineSteps, setPipelineSteps] = useState({});
  const [cleaning, setCleaning] = useState(false);
  const [cleanResult, setCleanResult] = useState(null);
  
  // History State
  const [runs, setRuns] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showPythonCode, setShowPythonCode] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
    setProfile(null);
    setCleanResult(null);
    setPipelineSteps({});
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first.");
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Upload file
      const uploadRes = await axios.post('http://localhost:5000/api/datasets', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const newDataset = uploadRes.data;
      setDataset(newDataset);
      
      // Fetch profile
      const profileRes = await axios.get(`http://localhost:5000/api/datasets/${newDataset._id}/profile`);
      setProfile(profileRes.data);
      
      // Auto-apply suggested fixes
      autoApplySuggestions(profileRes.data.columns);

    } catch (err) {
      console.error(err);
      setError(err.response?.data?.message || "An error occurred during upload/profiling.");
    } finally {
      setUploading(false);
    }
  };
  
  const autoApplySuggestions = (columns) => {
    const suggestedSteps = {};
    columns.forEach(col => {
      if (col.suggested && col.suggested.length > 0) {
        const firstSuggestion = col.suggested[0];
        if (firstSuggestion === "impute_median") {
          suggestedSteps[col.name] = { column: col.name, action: "impute", method: "median" };
        } else if (firstSuggestion === "impute_mode") {
          suggestedSteps[col.name] = { column: col.name, action: "impute", method: "mode" };
        } else if (firstSuggestion === "robust_scale") {
          suggestedSteps[col.name] = { column: col.name, action: "scale", method: "standard" };
        }
      }
    });
    setPipelineSteps(suggestedSteps);
  };
  
  useEffect(() => {
    fetchRuns();
  }, []);

  const handleActionChange = (colName, value) => {
    if (!value) {
      const newSteps = { ...pipelineSteps };
      delete newSteps[colName];
      setPipelineSteps(newSteps);
      return;
    }

    // Parse value like "impute:mean"
    const [action, method] = value.split(':');
    setPipelineSteps({
      ...pipelineSteps,
      [colName]: { column: colName, action, method }
    });
  };

  const handleClean = async () => {
    if (!dataset) return;
    
    setCleaning(true);
    setError(null);

    const steps = Object.values(pipelineSteps);

    try {
      const res = await axios.post(`http://localhost:5000/api/datasets/${dataset._id}/prepare`, {
        steps
      });
      setCleanResult(res.data);
      fetchRuns(); // Refresh history
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.message || "An error occurred during cleaning.");
    } finally {
      setCleaning(false);
    }
  };
  
  const fetchRuns = async () => {
    try {
      const res = await axios.get('http://localhost:5000/api/datasets/runs');
      setRuns(res.data);
    } catch (err) {
      console.error("Error fetching runs:", err);
    }
  };
  
  const downloadFile = (runId) => {
    window.open(`http://localhost:5000/api/datasets/runs/${runId}/download`, '_blank');
  };
  
  const copyPythonCode = () => {
    if (cleanResult?.python_code) {
      navigator.clipboard.writeText(cleanResult.python_code);
      alert("Python code copied to clipboard!");
    }
  };

  return (
    <div className="container">
      <header className="app-header">
        <h1>Autonomous Data Preparation Engine</h1>
        <button 
          className="history-btn" 
          onClick={() => setShowHistory(!showHistory)}
        >
          {showHistory ? 'Hide History' : 'View History'} ({runs.length})
        </button>
      </header>
      
      {showHistory && (
        <div className="history-section">
          <h2>Pipeline Run History</h2>
          <div className="history-list">
            {runs.map(run => (
              <div key={run._id} className="history-item">
                <div>
                  <strong>{run.dataset_id?.name || 'Unknown'}</strong>
                  <p>{new Date(run.created_at).toLocaleString()}</p>
                  <p className="small-text">{run.transformation_log.length} transformations</p>
                </div>
                <button onClick={() => downloadFile(run._id)} className="btn-small">
                  Download
                </button>
              </div>
            ))}
            {runs.length === 0 && <p>No runs yet</p>}
          </div>
        </div>
      )}
      
      <div className="upload-section">
        <input type="file" onChange={handleFileChange} accept=".csv,.xlsx,.xls" />
        <button onClick={handleUpload} disabled={uploading || !file}>
          {uploading ? 'Processing...' : 'Upload & Analyze'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {profile && (
        <div className="profile-section">
          <h2>Data Profile</h2>
          <div className="stats">
            <p><strong>Rows:</strong> {profile.rows}</p>
            <p><strong>Columns:</strong> {profile.columns.length}</p>
          </div>

          <h3>Column Analysis & Pipeline Builder</h3>
          <div className="columns-grid">
            {profile.columns.map((col) => (
              <div key={col.name} className={`column-card ${col.issues?.length > 0 ? 'has-issues' : ''}`}>
                <div className="column-header">
                  <h4>{col.name}</h4>
                  <span className="badge">{col.inferred_type}</span>
                </div>
                
                {col.issues && col.issues.length > 0 && (
                  <div className="issues-badge">
                    ⚠️ Issues: {col.issues.join(', ')}
                  </div>
                )}
                
                <div className="col-stats">
                  <p>Missing: {col.missing_pct.toFixed(1)}%</p>
                  <p>Unique: {col.unique}</p>
                  {col.inferred_type === 'numeric' && (
                    <div className="numeric-stats">
                      <small>Min: {col.min}</small>
                      <small>Max: {col.max}</small>
                      <small>Mean: {col.mean?.toFixed(2)}</small>
                      {col.outliers_count > 0 && (
                        <small className="warning">⚠️ Outliers: {col.outliers_count}</small>
                      )}
                    </div>
                  )}
                </div>

                <select 
                  className="action-select"
                  onChange={(e) => handleActionChange(col.name, e.target.value)}
                  value={pipelineSteps[col.name] ? `${pipelineSteps[col.name].action}:${pipelineSteps[col.name].method || ''}` : ""}
                >
                  <option value="">Select Action...</option>
                  <option value="drop:">Drop Column</option>
                  {col.inferred_type === 'numeric' && (
                    <>
                      <option value="impute:mean">Impute Mean</option>
                      <option value="impute:median">Impute Median</option>
                      <option value="scale:standard">Standard Scale</option>
                      <option value="scale:minmax">MinMax Scale</option>
                    </>
                  )}
                  {(col.inferred_type === 'text' || col.inferred_type === 'object') && (
                    <>
                      <option value="impute:mode">Impute Mode</option>
                      <option value="encode:onehot">One-Hot Encode</option>
                      <option value="encode:label">Label Encode</option>
                    </>
                  )}
                </select>
              </div>
            ))}
          </div>

          <div style={{textAlign: 'center'}}>
            <button className="clean-button" onClick={handleClean} disabled={cleaning}>
              {cleaning ? 'Running Pipeline...' : 'Run Cleaning Pipeline'}
            </button>
          </div>

          {cleanResult && (
            <div className="results-section">
              <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px'}}>
                <h2>Cleaning Results</h2>
                <div style={{display: 'flex', gap: '10px'}}>
                  <button 
                    className="download-btn"
                    onClick={() => downloadFile(cleanResult.run_id)}
                  >
                    Download Cleaned Dataset
                  </button>
                  <button 
                    className="code-btn"
                    onClick={() => setShowPythonCode(!showPythonCode)}
                  >
                    {showPythonCode ? 'Hide' : 'Show'} Python Code
                  </button>
                </div>
              </div>
              
              {showPythonCode && cleanResult.python_code && (
                <div className="python-code-section">
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                    <h3>Reproducible Python Code</h3>
                    <button className="copy-btn" onClick={copyPythonCode}>
                      Copy Code
                    </button>
                  </div>
                  <pre className="code-block">
                    <code>{cleanResult.python_code}</code>
                  </pre>
                </div>
              )}
              
              <p><strong>Cleaned File:</strong> {cleanResult.cleaned_file_path}</p>
              
              <h3>Transformation Log</h3>
              <div className="log-list">
                {cleanResult.transformation_log.map((log, i) => (
                  <div key={i} className="log-item">✅ {log}</div>
                ))}
                {cleanResult.transformation_log.length === 0 && <p>No transformations applied.</p>}
              </div>

              <h3>Preview (Cleaned)</h3>
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      {cleanResult.columns.map(col => <th key={col}>{col}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {cleanResult.preview.map((row, i) => (
                      <tr key={i}>
                        {cleanResult.columns.map(col => <td key={col}>{row[col] !== null ? row[col].toString() : ''}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {!cleanResult && (
            <>
              <h3>Preview (Raw)</h3>
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      {profile.columns.map(col => <th key={col.name}>{col.name}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {profile.preview.map((row, i) => (
                      <tr key={i}>
                        {profile.columns.map(col => <td key={col.name}>{row[col.name]}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
