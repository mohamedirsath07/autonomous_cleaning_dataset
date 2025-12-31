import React, { useState, useEffect } from 'react';
import { X, Clock, FileText, Download, Calendar, Activity } from 'lucide-react';
import axios from 'axios';

// API URL
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

const HistoryModal = ({ isOpen, onClose, user }) => {
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isOpen && user) {
      fetchHistory();
    }
  }, [isOpen, user]);

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(`${API_URL}/api/datasets/history/${user._id}`);
      setHistory(response.data);
    } catch (error) {
      console.error("Error fetching history:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-[#0a0a0a] border border-white/10 rounded-2xl w-full max-w-4xl h-[80vh] flex flex-col relative shadow-2xl shadow-[#ccff00]/10 animate-in fade-in zoom-in duration-200">
        
        {/* Header */}
        <div className="p-6 border-b border-white/10 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-[#ccff00]/10 rounded-lg text-[#ccff00]">
              <Clock size={24} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Activity History</h2>
              <p className="text-gray-400 text-sm">Your past cleaning operations</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-2 hover:bg-white/5 rounded-lg"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-thin scrollbar-thumb-white/10">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-3">
              <Activity className="animate-spin" size={32} />
              <p>Loading history...</p>
            </div>
          ) : history.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-3">
              <Clock size={48} className="opacity-20" />
              <p>No history found. Start cleaning some data!</p>
            </div>
          ) : (
            history.map((run) => (
              <div key={run._id} className="bg-white/5 border border-white/5 rounded-xl p-5 hover:border-[#ccff00]/30 transition-all group">
                <div className="flex justify-between items-start mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-500/10 text-blue-400 rounded-lg">
                      <FileText size={18} />
                    </div>
                    <div>
                      <h3 className="font-bold text-white">{run.dataset_id?.name || 'Unknown Dataset'}</h3>
                      <div className="flex items-center gap-2 text-xs text-gray-400 mt-1">
                        <Calendar size={12} />
                        {new Date(run.created_at).toLocaleString()}
                      </div>
                    </div>
                  </div>
                  
                  {run.cleaned_file_path && (
                    <a 
                      href={`${API_URL}${run.cleaned_file_path}`}
                      download
                      className="flex items-center gap-2 px-3 py-1.5 bg-[#ccff00]/10 text-[#ccff00] text-xs font-bold rounded-lg hover:bg-[#ccff00] hover:text-black transition-all"
                    >
                      <Download size={14} /> Download
                    </a>
                  )}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs text-gray-400 bg-black/20 p-3 rounded-lg">
                  <div>
                    <span className="block text-gray-500 mb-1">Split Ratio</span>
                    <span className="text-white">{run.pipeline_config?.splitRatio || 0}%</span>
                  </div>
                  <div>
                    <span className="block text-gray-500 mb-1">Outliers</span>
                    <span className="text-white">{run.pipeline_config?.removeOutliers ? 'Removed' : 'Kept'}</span>
                  </div>
                  <div>
                    <span className="block text-gray-500 mb-1">Imputation</span>
                    <span className="text-white">{run.pipeline_config?.imputeMissing ? 'Applied' : 'None'}</span>
                  </div>
                  <div>
                    <span className="block text-gray-500 mb-1">Normalization</span>
                    <span className="text-white">{run.pipeline_config?.normalizeFeatures ? 'Applied' : 'None'}</span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default HistoryModal;