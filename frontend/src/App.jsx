import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Upload,
  Settings,
  Database,
  Zap,
  LayoutTemplate,
  GitBranch,
  Terminal as TerminalIcon,
  CheckCircle,
  Download,
  BarChart,
  BrainCircuit,
  Layers,
  FileCheck,
  Activity
} from 'lucide-react';

// API URL - uses environment variable in production, localhost in development
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

const AutoKlean = () => {
  const [scrollY, setScrollY] = useState(0);
  const [activeTab, setActiveTab] = useState('config');

  // Data States
  const [file, setFile] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [profile, setProfile] = useState(null);
  const [cleanedFileUrl, setCleanedFileUrl] = useState(null);

  // Configuration States
  const [splitRatio, setSplitRatio] = useState(0);
  const [epochs, setEpochs] = useState(0);
  const [kFolds, setKFolds] = useState(0);
  const [removeOutliers, setRemoveOutliers] = useState(false);
  const [imputeMissing, setImputeMissing] = useState(false);
  const [normalizeFeatures, setNormalizeFeatures] = useState(false);
  const [generateSplitData, setGenerateSplitData] = useState(false);

  // Split Data ZIP URL
  const [splitZipUrl, setSplitZipUrl] = useState(null);

  // Processing States
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);
  const [pipelineResults, setPipelineResults] = useState(null);

  // Large file warning state
  const [showLargeFileWarning, setShowLargeFileWarning] = useState(false);
  const [pendingFile, setPendingFile] = useState(null);
  const MAX_FILE_SIZE_MB = 50; // 50MB warning threshold (Render free tier has 512MB RAM)

  // Handle Scroll
  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Handle File Upload
  const handleFileChange = async (e) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    // Check file size and show warning for large files
    const fileSizeMB = selectedFile.size / (1024 * 1024);
    if (fileSizeMB > MAX_FILE_SIZE_MB) {
      setPendingFile(selectedFile);
      setShowLargeFileWarning(true);
      return;
    }

    await processFileUpload(selectedFile);
  };

  // Process the file upload (called directly or after warning confirmation)
  const processFileUpload = async (selectedFile) => {
    setFile(selectedFile);
    setDataset(null);
    setProfile(null);
    setCleanedFileUrl(null);
    setSplitZipUrl(null);
    setPipelineResults(null);
    setShowLargeFileWarning(false);
    setPendingFile(null);
    setLogs(["Uploading and profiling dataset..."]);
    setIsProcessing(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/api/datasets/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setDataset(response.data);
      setProfile(response.data.profile);
      setLogs(prev => [...prev, "Upload complete.", "Profile generated successfully."]);
    } catch (error) {
      console.error("Upload error:", error);
      setLogs(prev => [...prev, "Error uploading file. Please try again."]);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle large file warning confirmation
  const handleLargeFileConfirm = () => {
    if (pendingFile) {
      processFileUpload(pendingFile);
    }
  };

  // Handle large file warning cancellation
  const handleLargeFileCancel = () => {
    setShowLargeFileWarning(false);
    setPendingFile(null);
  };

  // Run Pipeline
  const startProcessing = async () => {
    if (!dataset) return;

    setIsProcessing(true);
    setLogs(prev => [...prev, "Initializing pipeline...", "Sending configuration to ML service..."]);
    setCleanedFileUrl(null);
    setSplitZipUrl(null);
    setPipelineResults(null);

    try {
      const response = await axios.post(`${API_URL}/api/datasets/${dataset._id}/clean`, {
        splitRatio: generateSplitData ? splitRatio : 0,
        kFolds,
        epochs,
        removeOutliers,
        imputeMissing,
        normalizeFeatures
      });

      // Store full pipeline results
      setPipelineResults(response.data);
      console.log("Pipeline response:", response.data);

      // Update logs with the transformation log from backend
      if (response.data.transformation_log) {
        setLogs(prev => [...prev, ...response.data.transformation_log, "Pipeline completed successfully."]);
      } else {
        setLogs(prev => [...prev, "Pipeline completed."]);
      }

      if (response.data.cleanedFilePath) {
        setCleanedFileUrl(`${API_URL}${response.data.cleanedFilePath}`);
      }

      // Set train/test ZIP URL if available
      if (response.data.splitZipPath) {
        console.log("Setting splitZipUrl:", `${API_URL}${response.data.splitZipPath}`);
        setSplitZipUrl(`${API_URL}${response.data.splitZipPath}`);
      } else {
        console.log("No splitZipPath in response");
      }

    } catch (error) {
      console.error("Pipeline error:", error);
      setLogs(prev => [...prev, "Error executing pipeline.", error.message]);
    } finally {
      setIsProcessing(false);
    }
  };

  // --- Components ---

  const Reveal = ({ children }) => {
    // No animation - content is always visible immediately
    return <>{children}</>;
  };

  return (
    <div className="bg-[#030303] text-white min-h-screen font-sans selection:bg-[#ccff00] selection:text-black overflow-x-hidden">

      {/* Large File Warning Modal */}
      {showLargeFileWarning && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[100] flex items-center justify-center p-4">
          <div className="bg-[#0a0a0a] border border-yellow-500/30 rounded-2xl p-6 max-w-md w-full shadow-2xl">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-yellow-500/20 rounded-full flex items-center justify-center">
                <svg className="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <h3 className="text-lg font-bold text-yellow-500">Large File Detected</h3>
            </div>

            <p className="text-gray-300 text-sm mb-4">
              This file is <span className="text-white font-semibold">{pendingFile ? (pendingFile.size / (1024 * 1024)).toFixed(1) : 0} MB</span>, which may cause issues on the free hosting tier due to memory limitations.
            </p>

            <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-3 mb-6">
              <p className="text-xs text-yellow-400/90">
                <strong>Recommended:</strong> For best results on the free tier, use datasets under 50MB (~50,000 rows). Larger files may timeout or fail.
              </p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleLargeFileCancel}
                className="flex-1 py-3 px-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleLargeFileConfirm}
                className="flex-1 py-3 px-4 bg-yellow-500 hover:bg-yellow-400 text-black rounded-lg text-sm font-bold transition-colors"
              >
                Proceed Anyway
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Navbar */}
      <nav className={`fixed top-0 w-full z-50 px-6 py-4 transition-all duration-300 ${scrollY > 20 ? 'bg-[#030303]/90 backdrop-blur-md border-b border-white/5' : ''}`}>
        <div className="w-full max-w-[1600px] mx-auto flex justify-between items-center">
          <div className="flex items-center gap-2 font-bold text-xl tracking-tighter cursor-pointer" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
            <div className="w-4 h-4 bg-[#ccff00] rounded-sm transform rotate-45" />
            AUTOKLEAN
          </div>
          <div className="flex items-center gap-6">
            <button className="text-sm font-medium text-gray-400 hover:text-white transition-colors">Docs</button>
            <span className="text-xs text-gray-500 font-mono">v2.0</span>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6 flex flex-col items-center justify-center min-h-[90vh]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-white/5 via-transparent to-transparent opacity-50 blur-3xl pointer-events-none" />

        <div className="text-center max-w-6xl mx-auto space-y-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[#ccff00]/20 bg-[#ccff00]/5 text-[#ccff00] text-xs font-mono uppercase tracking-widest mb-4">
            <Zap size={12} /> V2.0 Auto-Pipeline Engine
          </div>

          <h1 className="text-6xl md:text-8xl lg:text-9xl font-bold tracking-tighter leading-[0.95]">
            Refine Data. <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-b from-white to-gray-500">Amplify Models.</span>
          </h1>

          <p className="text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed">
            Upload raw datasets and let AutoKlean handle missing values, normalization, splits, and feature engineering instantly.
          </p>

          {/* Upload Area */}
          <div className="mt-12 w-full max-w-2xl mx-auto group cursor-pointer relative">
            <input
              type="file"
              accept=".csv,.json"
              onChange={handleFileChange}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer block w-full h-full">
              <div className="absolute inset-0 bg-[#ccff00] opacity-0 group-hover:opacity-5 blur-2xl transition-opacity duration-500 rounded-2xl" />
              <div className="relative border-2 border-dashed border-white/10 bg-[#0a0a0a] rounded-2xl p-12 flex flex-col items-center justify-center gap-4 transition-all duration-300 group-hover:border-[#ccff00]/50 group-hover:scale-[1.01]">
                <div className="w-16 h-16 bg-[#1a1a1a] rounded-full flex items-center justify-center text-gray-400 group-hover:text-[#ccff00] transition-colors">
                  <Upload size={32} />
                </div>
                <div className="text-center">
                  <p className="text-lg font-bold text-white">
                    {file ? file.name : 'Drop your .CSV or .JSON here'}
                  </p>
                  <p className="text-sm text-gray-500">Supports Pandas, NumPy, and SQL exports</p>
                </div>
              </div>
            </label>
          </div>
        </div>
      </section>

      {/* Main Interface / Workspace */}
      <section className="py-20 px-4 md:px-8 max-w-[1600px] mx-auto">
        <Reveal>
          <div className="flex flex-col md:flex-row gap-8">

            {/* Left: Configuration Panel */}
            <div className="w-full md:w-1/3 space-y-6">
              <div className="bg-[#0a0a0a] border border-white/10 rounded-xl p-6 shadow-2xl sticky top-24">
                <div className="flex items-center gap-3 mb-6 border-b border-white/5 pb-4">
                  <Settings className="text-[#ccff00]" size={20} />
                  <h3 className="font-bold text-lg">Pipeline Config</h3>
                </div>

                {/* Train/Test Split */}
                <div className="space-y-4 mb-8">
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-gray-400">Train / Test Split</span>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min="0"
                        max="100"
                        value={splitRatio}
                        onChange={(e) => setSplitRatio(Math.max(0, Math.min(100, parseInt(e.target.value) || 0)))}
                        className="bg-white/5 border border-white/10 rounded px-2 py-1 w-16 text-center font-mono text-sm focus:outline-none focus:border-[#ccff00]"
                      />
                      <span className="text-[#ccff00] font-mono text-xs">% train</span>
                    </div>
                  </div>
                  <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="absolute left-0 top-0 h-full bg-[#ccff00] transition-all"
                      style={{ width: `${splitRatio}%` }}
                    />
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={splitRatio}
                      onChange={(e) => setSplitRatio(parseInt(e.target.value))}
                      className="absolute inset-0 w-full opacity-0 cursor-pointer"
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Train: {splitRatio}%</span>
                    <span>Test: {100 - splitRatio}%</span>
                  </div>
                </div>

                {/* K-Folds Input */}
                <div className="space-y-4 mb-8">
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-gray-400">Cross Validation (K-Folds)</span>
                    <div className="flex items-center gap-2">
                      <button onClick={() => setKFolds(Math.max(0, kFolds - 1))} className="w-6 h-6 rounded bg-white/5 hover:bg-white/10 flex items-center justify-center">-</button>
                      <input
                        type="number"
                        min="0"
                        max="20"
                        value={kFolds}
                        onChange={(e) => setKFolds(Math.max(0, Math.min(20, parseInt(e.target.value) || 0)))}
                        className="bg-white/5 border border-white/10 rounded px-2 py-1 w-12 text-center font-mono text-sm focus:outline-none focus:border-[#ccff00]"
                      />
                      <button onClick={() => setKFolds(Math.min(20, kFolds + 1))} className="w-6 h-6 rounded bg-white/5 hover:bg-white/10 flex items-center justify-center">+</button>
                    </div>
                  </div>
                </div>

                {/* Epochs Input */}
                <div className="space-y-4 mb-8">
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-gray-400">Target Epochs</span>
                    <input
                      type="number"
                      value={epochs}
                      onChange={(e) => setEpochs(e.target.value)}
                      className="bg-white/5 border border-white/10 rounded px-2 py-1 w-20 text-right font-mono text-sm focus:outline-none focus:border-[#ccff00]"
                    />
                  </div>
                </div>

                {/* Toggles */}
                <div className="space-y-3 mb-8">
                  {[
                    { label: 'Remove Outliers', state: removeOutliers, setState: setRemoveOutliers },
                    { label: 'Impute Missing (KNN)', state: imputeMissing, setState: setImputeMissing },
                    { label: 'Normalize Features', state: normalizeFeatures, setState: setNormalizeFeatures },
                    { label: 'Generate Train/Test Split', state: generateSplitData, setState: setGenerateSplitData }
                  ].map((item) => (
                    <div
                      key={item.label}
                      onClick={() => item.setState(!item.state)}
                      className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${item.state ? 'bg-white/5 border-[#ccff00]/30' : 'bg-transparent border-white/5 hover:border-white/20'}`}
                    >
                      <span className={`text-sm ${item.state ? 'text-white' : 'text-gray-300'}`}>{item.label}</span>
                      <div className={`w-4 h-4 rounded-full transition-all ${item.state ? 'bg-[#ccff00] shadow-[0_0_10px_rgba(204,255,0,0.5)]' : 'bg-gray-800'}`} />
                    </div>
                  ))}
                </div>

                <button
                  onClick={startProcessing}
                  disabled={isProcessing}
                  className={`w-full py-4 rounded-lg font-bold text-black transition-all ${isProcessing ? 'bg-gray-600 cursor-not-allowed' : 'bg-[#ccff00] hover:bg-[#b3e600] hover:scale-[1.02]'}`}
                >
                  {isProcessing ? 'PROCESSING...' : 'RUN PIPELINE'}
                </button>
              </div>
            </div>

            {/* Right: Visualization & Output */}
            <div className="w-full md:w-2/3 space-y-6">

              {/* Tab Navigation */}
              <div className="flex gap-4 border-b border-white/10 pb-1">
                <button
                  onClick={() => setActiveTab('config')}
                  className={`pb-3 text-sm font-medium transition-colors border-b-2 ${activeTab === 'config' ? 'border-[#ccff00] text-white' : 'border-transparent text-gray-500 hover:text-white'}`}
                >
                  Dataset Overview
                </button>
                <button
                  onClick={() => setActiveTab('features')}
                  className={`pb-3 text-sm font-medium transition-colors border-b-2 ${activeTab === 'features' ? 'border-[#ccff00] text-white' : 'border-transparent text-gray-500 hover:text-white'}`}
                >
                  Feature Engineering
                </button>
              </div>

              {/* Data Health Grid (Bento) */}
              {activeTab === 'config' && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="col-span-2 md:col-span-2 bg-[#0a0a0a] border border-white/10 p-6 rounded-xl relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                      <Database size={60} />
                    </div>
                    <div className="text-gray-500 text-xs uppercase font-mono mb-2">Total Records</div>
                    <div className="text-4xl font-bold text-white">
                      {profile?.shape?.[0]?.toLocaleString() || (dataset ? '...' : '0')}
                    </div>
                    <div className="text-emerald-500 text-xs mt-2 flex items-center gap-1">
                      <CheckCircle size={10} /> {dataset ? '100% Load Success' : 'No Data'}
                    </div>
                  </div>

                  <div className="col-span-1 bg-[#0a0a0a] border border-white/10 p-4 rounded-xl">
                    <div className="text-gray-500 text-xs uppercase font-mono mb-1">Missing</div>
                    <div className="text-2xl font-bold text-white">
                      {profile?.missing_values ?
                        ((Object.values(profile.missing_values).reduce((a, b) => a + b, 0) / (profile.shape[0] * profile.shape[1]) * 100).toFixed(1))
                        : '0'}%
                    </div>
                    <div className="w-full bg-gray-800 h-1 mt-3 rounded-full overflow-hidden">
                      <div
                        className="bg-red-500 h-full"
                        style={{ width: `${profile?.missing_values ? ((Object.values(profile.missing_values).reduce((a, b) => a + b, 0) / (profile.shape[0] * profile.shape[1]) * 100)) : 0}%` }}
                      />
                    </div>
                  </div>

                  <div className="col-span-1 bg-[#0a0a0a] border border-white/10 p-4 rounded-xl">
                    <div className="text-gray-500 text-xs uppercase font-mono mb-1">Columns</div>
                    <div className="text-2xl font-bold text-white">{profile?.shape?.[1] || '0'}</div>
                    <div className="text-xs text-gray-500 mt-2">Features Detected</div>
                  </div>

                  {/* Feature Distribution Graph Mockup */}
                  <div className="col-span-2 md:col-span-4 bg-[#0a0a0a] border border-white/10 p-6 rounded-xl h-48 flex flex-col justify-between">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-bold">Feature Distribution (Target Variable)</span>
                      <BarChart size={16} className="text-gray-500" />
                    </div>
                    <div className="flex items-end gap-2 h-24 w-full">
                      {[40, 65, 45, 90, 30, 60, 55, 80, 50, 75, 40, 95, 60, 45, 70, 85].map((h, i) => (
                        <div
                          key={i}
                          className="flex-1 bg-white/10 hover:bg-[#ccff00] transition-all duration-300 rounded-t-sm"
                          style={{ height: `${h}%` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Results Panel */}
              {(cleanedFileUrl || pipelineResults) && (
                <div className="bg-gradient-to-br from-[#0f0f0f] to-[#0a0a0a] rounded-xl border border-[#ccff00]/20 p-5 shadow-lg">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="text-[#ccff00]" size={18} />
                      <span className="font-bold text-white">Pipeline Results</span>
                    </div>
                    <span className="text-[10px] bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded-full uppercase tracking-wider">Success</span>
                  </div>

                  {/* Results Grid - Extended */}
                  <div className="grid grid-cols-3 md:grid-cols-6 gap-2 mb-4">
                    <div className="bg-black/40 rounded-lg p-3 border border-white/5">
                      <div className="flex items-center gap-1 text-[9px] text-gray-500 uppercase tracking-wider mb-1">
                        <GitBranch size={10} />Train
                      </div>
                      <div className="text-base font-bold text-[#ccff00]">{splitRatio}%</div>
                    </div>
                    <div className="bg-black/40 rounded-lg p-3 border border-white/5">
                      <div className="flex items-center gap-1 text-[9px] text-gray-500 uppercase tracking-wider mb-1">
                        <FileCheck size={10} />Test
                      </div>
                      <div className="text-base font-bold text-blue-400">{100 - splitRatio}%</div>
                    </div>
                    <div className="bg-black/40 rounded-lg p-3 border border-white/5">
                      <div className="flex items-center gap-1 text-[9px] text-gray-500 uppercase tracking-wider mb-1">
                        <Layers size={10} />K-Folds
                      </div>
                      <div className="text-base font-bold text-white">{kFolds}</div>
                    </div>
                    <div className="bg-black/40 rounded-lg p-3 border border-white/5">
                      <div className="flex items-center gap-1 text-[9px] text-gray-500 uppercase tracking-wider mb-1">
                        <Activity size={10} />Epochs
                      </div>
                      <div className="text-base font-bold text-white">{epochs}</div>
                    </div>
                    <div className="bg-black/40 rounded-lg p-3 border border-white/5">
                      <div className="flex items-center gap-1 text-[9px] text-gray-500 uppercase tracking-wider mb-1">
                        <Database size={10} />Rows
                      </div>
                      <div className="text-base font-bold text-white">{profile?.shape?.[0]?.toLocaleString() || '-'}</div>
                    </div>
                    <div className="bg-black/40 rounded-lg p-3 border border-white/5">
                      <div className="flex items-center gap-1 text-[9px] text-gray-500 uppercase tracking-wider mb-1">
                        <BarChart size={10} />Cols
                      </div>
                      <div className="text-base font-bold text-white">{profile?.shape?.[1] || '-'}</div>
                    </div>
                  </div>

                  {/* Processing Options Summary */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {removeOutliers && (
                      <span className="text-[10px] bg-white/5 text-gray-400 px-2 py-1 rounded border border-white/10">Outliers Removed</span>
                    )}
                    {imputeMissing && (
                      <span className="text-[10px] bg-white/5 text-gray-400 px-2 py-1 rounded border border-white/10">Missing Imputed (KNN)</span>
                    )}
                    {normalizeFeatures && (
                      <span className="text-[10px] bg-white/5 text-gray-400 px-2 py-1 rounded border border-white/10">Features Normalized</span>
                    )}
                    {generateSplitData && splitZipUrl && (
                      <span className="text-[10px] bg-[#ccff00]/10 text-[#ccff00] px-2 py-1 rounded border border-[#ccff00]/20">Train/Test Split Generated</span>
                    )}
                  </div>

                  {/* Download Buttons */}
                  <div className="space-y-3">
                    {/* Main Cleaned Dataset */}
                    {cleanedFileUrl && (
                      <a
                        href={cleanedFileUrl}
                        download
                        className="flex items-center justify-center gap-3 w-full py-4 bg-[#ccff00] text-black rounded-lg hover:bg-[#b3e600] transition-all font-bold text-sm uppercase tracking-wider hover:scale-[1.02] shadow-lg shadow-[#ccff00]/20"
                      >
                        <Download size={18} /> Download Cleaned Dataset
                      </a>
                    )}

                    {/* Train/Test Split ZIP Download */}
                    {splitZipUrl && (
                      <a
                        href={splitZipUrl}
                        download="train_test_split.zip"
                        className="flex items-center justify-center gap-2 w-full py-3 bg-gradient-to-r from-emerald-500/20 to-blue-500/20 text-white border border-white/10 rounded-lg hover:from-emerald-500/30 hover:to-blue-500/30 transition-all font-bold text-xs uppercase tracking-wider"
                      >
                        <Download size={14} /> Download Train/Test Split (ZIP) • {splitRatio}% / {100 - splitRatio}%
                      </a>
                    )}
                  </div>
                </div>
              )}

              {/* Terminal / Logs Output - Compact */}
              <div className="bg-[#0a0a0a] rounded-lg border border-white/5 p-3 font-mono shadow-inner">
                <div className="flex items-center gap-2 border-b border-white/5 pb-2 mb-2 text-gray-600">
                  <TerminalIcon size={12} />
                  <span className="text-[10px] uppercase tracking-wider">System Log</span>
                  {isProcessing && <span className="ml-auto text-[10px] text-[#ccff00] animate-pulse">● Processing</span>}
                </div>
                <div className="max-h-32 overflow-y-auto space-y-0.5 scrollbar-thin scrollbar-thumb-white/10">
                  {logs.length === 0 && !isProcessing && (
                    <span className="text-gray-700 text-[11px] italic">Ready to process...</span>
                  )}
                  {logs.map((log, i) => (
                    <div key={i} className="text-[11px] text-gray-500 flex items-start gap-1.5 leading-tight">
                      <span className="text-[#ccff00]/60 text-[10px]">›</span>
                      <span className="truncate">{typeof log === 'string' && log.length > 80 ? log.substring(0, 80) + '...' : log}</span>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          </div>
        </Reveal>
      </section>

      {/* Feature Engineering Showcase Section */}
      <section className="py-20 bg-[#050505] border-t border-white/5">
        <div className="max-w-[1600px] mx-auto px-6">
          <Reveal>
            <h2 className="text-3xl md:text-5xl font-bold mb-12 flex items-center gap-4">
              <BrainCircuit className="text-[#ccff00]" size={40} />
              Intelligent Transformations
            </h2>
          </Reveal>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { title: 'Auto-Scaling', icon: <BarChart />, desc: 'Automatically applies MinMax or StandardScaler based on distribution variance.' },
              { title: 'Smart Encoding', icon: <LayoutTemplate />, desc: 'Detects categorical columns and applies One-Hot or Label Encoding optimized for cardinality.' },
              { title: 'Data Splitting', icon: <GitBranch />, desc: 'Stratified sampling ensures your Test set represents the real world accurately.' },
            ].map((feature, i) => (
              <Reveal key={i} delay={i * 100}>
                <div className="p-8 border border-white/10 rounded-2xl bg-[#0a0a0a] hover:border-[#ccff00]/50 transition-colors group">
                  <div className="w-12 h-12 bg-white/5 rounded-lg flex items-center justify-center mb-6 text-white group-hover:text-[#ccff00] group-hover:bg-[#ccff00]/10 transition-all">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                  <p className="text-gray-400 leading-relaxed">{feature.desc}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-12 text-center text-gray-500 text-sm bg-black">
        <p>AUTOKLEAN v2.0 • AUTO-CLEANING ENGINE</p>
      </footer>
    </div>
  );
};

export default AutoKlean;
