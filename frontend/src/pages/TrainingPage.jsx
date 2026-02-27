import React, { useEffect, useState, useRef, useContext } from 'react';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import { ThemeContext } from '../components/context/ThemeContext';
import Footer from '../components/Footer';
import ChatbotWidget from './ChatbotWidget';
import { FaCogs, FaTerminal, FaHistory, FaBrain, FaPlay } from 'react-icons/fa';

const TrainingPage = () => {
  const { theme } = useContext(ThemeContext);
  const [models, setModels] = useState({});
  const [jobs, setJobs] = useState([]);
  const [selectedModel, setSelectedModel] = useState('ECG1DCNN');
  const [params, setParams] = useState({
    epochs: 30,
    lr: 0.001,
    batch: 64,
    use_ai_labels: false
  });
  const [activeJob, setActiveJob] = useState(null); // { job_id, status, logs }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const logEndRef = useRef(null);

  // Fetch models and jobs on mount
  useEffect(() => {
    API.get('/model_list/')
      .then(resp => setModels(resp.data))
      .catch(err => setError('Failed to load models.'));
    
    fetchJobs();
  }, []);

  const fetchJobs = () => {
    API.get('/train/jobs/')
      .then(resp => setJobs(resp.data))
      .catch(() => {});
  };

  // Poll active job status
  useEffect(() => {
    let interval;
    if (activeJob && activeJob.status === 'running') {
      interval = setInterval(() => {
        API.get(`/train/${activeJob.job_id}/status/`)
          .then(resp => {
            setActiveJob(resp.data);
            if (resp.data.status !== 'running') {
              clearInterval(interval);
              fetchJobs();
            }
          })
          .catch(() => clearInterval(interval));
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [activeJob]);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeJob?.logs]);

  const handleStartTraining = async () => {
    setLoading(true);
    setError('');
    try {
      const resp = await API.post('/train/', {
        model: selectedModel,
        ...params
      });
      setActiveJob({
        job_id: resp.data.job_id,
        status: 'running',
        logs: [resp.data.message]
      });
      fetchJobs();
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex flex-col bg-[var(--bg)] text-[var(--text)] transition-colors duration-300`}>
      <DashboardNavbar />
      
      <main className="flex-grow container mx-auto px-4 py-10 max-w-6xl">
        <header className="mb-10 text-center">
          <h1 className="text-3xl font-bold tracking-tight mb-2">Model Training Environment</h1>
          <p className="text-xs font-bold uppercase tracking-widest text-gray-500">Fine-tune Clinical Inference Engines</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Config */}
          <div className="lg:col-span-4 space-y-6">
            <section className="card p-6 border-[var(--border)]">
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-gray-500 mb-6">
                <FaCogs /> Parameter Configuration
              </div>
              
              <div className="space-y-5">
                <div>
                  <label className="text-[10px] font-bold uppercase text-gray-400 mb-1.5 block">Target Architecture</label>
                  <select 
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-4 py-3 text-xs font-bold focus:outline-none focus:ring-1 focus:ring-[var(--accent)] transition-all"
                  >
                    {Object.entries(models).map(([name, info]) => (
                      <option key={name} value={name}>{info.label}</option>
                    ))}
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-[10px] font-bold uppercase text-gray-400 mb-1.5 block">Epochs</label>
                    <input 
                      type="number"
                      value={params.epochs}
                      onChange={(e) => setParams({...params, epochs: parseInt(e.target.value)})}
                      className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-4 py-3 text-xs font-bold focus:outline-none focus:ring-1 focus:ring-[var(--accent)] transition-all"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] font-bold uppercase text-gray-400 mb-1.5 block">Batch Size</label>
                    <input 
                      type="number"
                      value={params.batch}
                      onChange={(e) => setParams({...params, batch: parseInt(e.target.value)})}
                      className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-4 py-3 text-xs font-bold focus:outline-none focus:ring-1 focus:ring-[var(--accent)] transition-all"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-[10px] font-bold uppercase text-gray-400 mb-1.5 block">Learning Rate</label>
                  <input 
                    type="number"
                    step="0.0001"
                    value={params.lr}
                    onChange={(e) => setParams({...params, lr: parseFloat(e.target.value)})}
                    className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-4 py-3 text-xs font-bold focus:outline-none focus:ring-1 focus:ring-[var(--accent)] transition-all"
                  />
                </div>

                <div className="flex items-center gap-3 pt-2">
                  <input 
                    type="checkbox"
                    id="use_ai"
                    checked={params.use_ai_labels}
                    onChange={(e) => setParams({...params, use_ai_labels: e.target.checked})}
                    className="w-4 h-4 rounded border-[var(--border)] bg-[var(--highlight)] text-[var(--accent)] focus:ring-[var(--accent)]"
                  />
                  <label htmlFor="use_ai" className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide cursor-pointer select-none">Include Silver Labels</label>
                </div>

                <button
                  onClick={handleStartTraining}
                  disabled={loading || (activeJob && activeJob.status === 'running')}
                  className={`w-full py-4 mt-4 rounded-xl text-white font-bold uppercase tracking-widest text-xs transition-all flex items-center justify-center gap-3 shadow-lg ${
                    loading || (activeJob && activeJob.status === 'running')
                    ? 'bg-gray-400 cursor-not-allowed opacity-50' 
                    : 'bg-[var(--accent)] hover:shadow-[var(--accent)]/20 active:scale-[0.98]'
                  }`}
                >
                  {loading ? (
                    <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div> INITIALIZING...</>
                  ) : activeJob?.status === 'running' ? (
                    'TRAINING IN PROGRESS'
                  ) : (
                    <><FaPlay size={12} /> DEPLOY TRAINING JOB</>
                  )}
                </button>
                {error && <p className="text-red-500 text-[10px] font-bold uppercase mt-2 text-center">{error}</p>}
              </div>
            </section>

            <section className="card p-6 border-[var(--border)]">
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-gray-500 mb-4">
                <FaHistory /> Session History
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto pr-2 custom-scrollbar">
                {jobs.length === 0 ? (
                  <p className="text-[10px] text-gray-400 italic uppercase tracking-widest text-center py-4">No recent sessions</p>
                ) : (
                  jobs.map(job => (
                    <div 
                      key={job.job_id}
                      onClick={() => setActiveJob({ ...job, logs: [] })} 
                      className="p-3 bg-[var(--highlight)] rounded-lg border border-[var(--border)] cursor-pointer hover:border-[var(--accent)] transition-all flex justify-between items-center group"
                    >
                      <div className="space-y-0.5">
                        <div className="text-[11px] font-bold uppercase">{job.model}</div>
                        <div className="text-[9px] text-gray-400 font-mono tracking-tighter">{job.job_id}</div>
                      </div>
                      <span className={`text-[9px] px-2 py-0.5 rounded-full uppercase font-bold border ${
                        job.status === 'done' ? 'bg-green-100/10 border-green-500/20 text-green-500' : 
                        job.status === 'running' ? 'bg-blue-100/10 border-blue-500/20 text-blue-500' : 'bg-red-100/10 border-red-500/20 text-red-500'
                      }`}>
                        {job.status}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </section>
          </div>

          {/* Right Column: Terminal */}
          <div className="lg:col-span-8 flex flex-col h-[700px]">
            <section className="bg-black/95 rounded-2xl shadow-2xl border border-[var(--border)] overflow-hidden flex flex-col h-full">
              <div className="bg-[#1a1a1a] px-5 py-3 flex items-center justify-between border-b border-white/5">
                <div className="flex items-center gap-3">
                  <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/50"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500/50"></div>
                  </div>
                  <div className="h-4 w-px bg-white/10 mx-1"></div>
                  <div className="flex items-center gap-2 text-[10px] font-bold text-gray-500 uppercase tracking-widest">
                    <FaTerminal /> Training Stream
                  </div>
                </div>
                <div className="text-[9px] font-mono text-gray-600 tracking-tight">
                  {activeJob ? `process::${activeJob.job_id}` : 'system::standby'}
                </div>
              </div>
              
              <div className="flex-grow p-6 font-mono text-xs overflow-y-auto custom-scrollbar bg-black selection:bg-[var(--accent)] selection:text-white">
                {!activeJob ? (
                  <div className="h-full flex flex-col items-center justify-center text-gray-700 gap-6 opacity-30">
                    <FaBrain size={48} />
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-center max-w-[200px] leading-relaxed">
                      Initialize session parameters to monitor real-time inference optimization
                    </p>
                  </div>
                ) : (
                  <div className="space-y-1.5">
                    {activeJob.logs?.map((line, i) => (
                      <div key={i} className="flex gap-4 group">
                        <span className="text-gray-800 select-none min-w-[20px] text-right font-bold">{i+1}</span>
                        <span className={
                          line.includes('ERROR') ? 'text-red-500' : 
                          line.includes('BEST') ? 'text-[var(--accent)] font-bold' : 
                          line.includes('complete') ? 'text-green-500 font-bold' : 
                          line.includes('Epoch') ? 'text-blue-400' : 'text-gray-400'
                        }>
                          {line}
                        </span>
                      </div>
                    ))}
                    <div ref={logEndRef} />
                  </div>
                )}
              </div>

              {activeJob?.status === 'running' && (
                <div className="bg-[#0a0a0a] px-6 py-4 border-t border-white/5 flex items-center gap-4">
                  <div className="flex-grow bg-white/5 h-1 rounded-full overflow-hidden">
                    <div className="bg-[var(--accent)] h-full w-2/3 animate-[shimmer_2s_infinite]"></div>
                  </div>
                  <span className="text-[10px] text-[var(--accent)] font-bold tracking-[0.2em] animate-pulse">OPTIMIZING_WEIGHTS</span>
                </div>
              )}
            </section>
          </div>
        </div>
      </main>

      <Footer />
      <ChatbotWidget />
      
      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
        
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
};

export default TrainingPage;
