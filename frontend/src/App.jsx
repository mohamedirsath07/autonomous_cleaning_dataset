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
      
      // Scroll to results
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
      
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
    <div className="app-container">
      <header className="app-header">
        <div className="logo-area">
          <span className="logo-icon">✨</span>
          <span className="app-title">DataForge</span>
        </div>
        <button 
          className="history-toggle" 
          onClick={() => setShowHistory(!showHistory)}
        >
          📜 History ({runs.length})
        </button>
      </header>
      
      <div className={`history-sidebar ${showHistory ? 'open' : ''}`}>
        <button className="close-btn" onClick={() => setShowHistory(false)}>&times;</button>
        <h2>Pipeline History</h2>
        <div className="history-list">
          {runs.map(run => (
            <div key={run._id} className="history-item">
              <div>
                <strong>{run.dataset_id?.name || 'Unknown Dataset'}</strong>
                <p style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>{new Date(run.created_at).toLocaleString()}</p>
                <p className="small-text">{run.transformation_log.length} transformations</p>
              </div>
              <button 
                onClick={() => downloadFile(run._id)} 
                className="btn-secondary"
                style={{marginTop: '0.5rem', width: '100%', fontSize: '0.8rem'}}
              >
                Download Result
              </button>
            </div>
          ))}
          {runs.length === 0 && <p style={{color: 'var(--text-secondary)', textAlign: 'center'}}>No runs yet.</p>}
        </div>
      </div>
      
      {!profile && !cleanResult && (
        <div className="hero-section fade-in">
          <h1 className="hero-title">Autonomous Data Cleaning</h1>
          <p className="hero-subtitle">
            Upload your messy dataset and let our AI-powered engine profile, clean, and generate production-ready Python pipelines for you.
          </p>
        </div>
      )}

      <div className="upload-section fade-in">
        {!profile ? (
          <div className="upload-card">
            <div className="file-input-wrapper">
              <input 
                type="file" 
                className="file-input" 
                onChange={handleFileChange} 
                accept=".csv,.xlsx,.xls" 
              />
              <div style={{pointerEvents: 'none'}}>
                <div style={{fontSize: '3rem', marginBottom: '1rem'}}>📂</div>
                <h3 style={{marginBottom: '0.5rem', color: 'var(--text-main)'}}>
                  {file ? file.name : "Drag & Drop or Click to Upload"}
                </h3>
                <p style={{color: 'var(--text-secondary)'}}>Supports CSV and Excel files</p>
              </div>
            </div>
            
            {file && (
              <div style={{marginTop: '2rem'}}>
                <button 
                  className="upload-btn" 
                  onClick={handleUpload} 
                  disabled={uploading}
                >
                  {uploading ? 'Analyzing Dataset...' : 'Start Profiling 🚀'}
                </button>
              </div>
            )}
          </div>
        ) : (
          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem'}}>
            <div>
              <h2 style={{margin: 0, color: 'var(--text-main)'}}>Current Dataset: {dataset?.name}</h2>
              <p style={{color: 'var(--text-secondary)'}}>Ready for pipeline configuration</p>
            </div>
            <button 
              className="btn-secondary"
              onClick={() => {
                setProfile(null);
                setDataset(null);
                setFile(null);
                setCleanResult(null);
              }}
            >
              Upload New File
            </button>
          </div>
        )}
      </div>

      {error && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)', 
          color: '#ef4444', 
          padding: '1rem', 
          borderRadius: '0.5rem', 
          marginBottom: '2rem',
          border: '1px solid #ef4444'
        }}>
          🚨 {error}
        </div>
      )}

      {profile && (
        <div className="profile-section fade-in">
          <div className="section-header">
            <h2 className="section-title">Data Profile & Pipeline Builder</h2>
            <div className="stats-bar">
              <div className="stat-item">Rows <span className="stat-value">{profile.rows}</span></div>
              <div className="stat-item">Columns <span className="stat-value">{profile.columns.length}</span></div>
            </div>
          </div>

          <div className="columns-grid">
            {profile.columns.map((col) => (
              <div key={col.name} className={`column-card ${col.issues?.length > 0 ? 'has-issues' : ''}`}>
                <div className="col-header">
                  <span className="col-name">{col.name}</span>
                  <span className="col-type">{col.inferred_type}</span>
                </div>
                
                {col.issues && col.issues.length > 0 && (
                  <div className="issue-tag">
                    ⚠️ {col.issues[0]} {col.issues.length > 1 && `+${col.issues.length - 1} more`}
                  </div>
                )}
                
                <div className="col-details">
                  <div className="col-detail-row">
                    <span>Missing</span>
                    <strong>{col.missing_pct.toFixed(1)}%</strong>
                  </div>
                  <div className="col-detail-row">
                    <span>Unique</span>
                    <strong>{col.unique}</strong>
                  </div>
                  {col.inferred_type === 'numeric' && (
                    <>
                      <div className="col-detail-row">
                        <span>Mean</span>
                        <strong>{col.mean?.toFixed(2)}</strong>
                      </div>
                      {col.outliers_count > 0 && (
                        <div className="col-detail-row" style={{color: '#f59e0b'}}>
                          <span>Outliers</span>
                          <strong>{col.outliers_count}</strong>
                        </div>
                      )}
                    </>
                  )}
                </div>

                <select 
                  className="action-select"
                  onChange={(e) => handleActionChange(col.name, e.target.value)}
                  value={pipelineSteps[col.name] ? `${pipelineSteps[col.name].action}:${pipelineSteps[col.name].method || ''}` : ""}
                >
                  <option value="">No Action</option>
                  <option value="drop:">🗑️ Drop Column</option>
                  {col.inferred_type === 'numeric' && (
                    <>
                      <option value="impute:mean">📊 Impute Mean</option>
                      <option value="impute:median">📊 Impute Median</option>
                      <option value="scale:standard">📏 Standard Scale</option>
                      <option value="scale:minmax">📏 MinMax Scale</option>
                    </>
                  )}
                  {(col.inferred_type === 'text' || col.inferred_type === 'object') && (
                    <>
                      <option value="impute:mode">📊 Impute Mode</option>
                      <option value="encode:onehot">🔠 One-Hot Encode</option>
                      <option value="encode:label">🏷️ Label Encode</option>
                    </>
                  )}
                </select>
              </div>
            ))}
          </div>

          <div className="action-bar fade-in">
            <button className="run-btn" onClick={handleClean} disabled={cleaning}>
              {cleaning ? 'Running Pipeline...' : ' Run Cleaning Pipeline'}
            </button>
          </div>

          {cleanResult && (
            <div id="results-section" className="results-container">
              <div className="results-header">
                <div>
                  <h2 style={{margin: 0}}>🎉 Cleaning Complete!</h2>
                  <p style={{color: '#666'}}>Your data has been transformed successfully.</p>
                </div>
                <div className="btn-group">
                  <button 
                    className="btn-primary"
                    onClick={() => downloadFile(cleanResult.run_id)}
                  >
                    📥 Download Cleaned Data
                  </button>
                  <button 
                    className="btn-secondary"
                    onClick={() => setShowPythonCode(!showPythonCode)}
                  >
                    {showPythonCode ? 'Hide Code' : 'Show Python Code'}
                  </button>
                </div>
              </div>
              
              {showPythonCode && cleanResult.python_code && (
                <div className="fade-in">
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem'}}>
                    <h3>Generated Python Pipeline</h3>
                    <button className="btn-secondary" style={{fontSize: '0.8rem'}} onClick={copyPythonCode}>
                      📋 Copy
                    </button>
                  </div>
                  <pre className="code-block">
                    <code>{cleanResult.python_code}</code>
                  </pre>
                </div>
              )}

              {cleanResult.preview && cleanResult.preview.length > 0 && (
                <div className="fade-in" style={{marginTop: '2rem'}}>
                  <h3>Cleaned Data Preview (First 20 Rows)</h3>
                  <div className="table-container">
                    <table className="data-table">
                      <thead>
                        <tr>
                          {cleanResult.columns.map((col) => (
                            <th key={col}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {cleanResult.preview.map((row, idx) => (
                          <tr key={idx}>
                            {cleanResult.columns.map((col) => (
                              <td key={`${idx}-${col}`}>
                                {row[col] !== null ? String(row[col]) : <span style={{color: '#ccc'}}>null</span>}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              
              <h3>Transformation Log</h3>
              <div className="log-list">
                {cleanResult.transformation_log.map((log, i) => (
                  <div key={i} className="log-item">
                    <span style={{color: '#10b981'}}>✓</span> {log}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
