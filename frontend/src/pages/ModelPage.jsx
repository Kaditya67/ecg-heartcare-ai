import React, { useState, useEffect, useCallback, useContext } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import Footer from '../components/Footer';
import { ThemeContext } from '../components/context/ThemeContext';
import { FaCheckCircle, FaTimesCircle, FaBrain, FaCogs, FaMicrochip } from 'react-icons/fa';

const LoadingOverlay = ({ loading, children, text = "Loading..." }) => (
  <div className="relative">
    {children}
    {loading && (
      <div className="absolute inset-0 bg-[var(--bg)]/50 backdrop-blur-[1px] z-50 flex items-center justify-center rounded-lg">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-4 border-[var(--accent)] border-t-transparent rounded-full animate-spin"></div>
          <span className="text-[10px] font-bold text-[var(--accent)] uppercase tracking-widest">{text}</span>
        </div>
      </div>
    )}
  </div>
);

const ModelPage = () => {
  const { theme } = useContext(ThemeContext);
  const [modelRegistry, setModelRegistry] = useState({});
  const [loading, setLoading] = useState(false);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await API.get('/model_list/');
      setModelRegistry(resp.data);
    } catch (err) {
      toast.error('Failed to fetch local model registry.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  return (
    <div className="min-h-screen flex flex-col bg-[var(--bg)] text-[var(--text)] transition-colors duration-300">
      <DashboardNavbar />
      
      <main className="flex-grow p-4 lg:p-10">
        <div className="max-w-[1000px] mx-auto space-y-10">
          
          <header className="flex flex-col items-center text-center space-y-2 border-b border-[var(--border)] pb-8">
            <h1 className="text-3xl font-bold tracking-tight">Intelligence Inventory</h1>
            <p className="text-xs font-bold uppercase tracking-widest text-gray-500">Local Neural Asset Management</p>
          </header>

          <section className="space-y-6">
            <div className="flex items-center justify-between px-2">
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-gray-500">
                <FaCogs /> Registered Inference Engines
              </div>
              <button 
                onClick={fetchModels}
                className="text-[10px] font-bold text-[var(--accent)] hover:underline uppercase tracking-widest"
              >
                Refresh Registry
              </button>
            </div>

            <LoadingOverlay loading={loading} text="Auditing Local Vault...">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {Object.entries(modelRegistry).map(([id, info]) => (
                  <div key={id} className="card group hover:border-[var(--accent)] transition-all flex flex-col">
                    <div className="flex justify-between items-start mb-4">
                      <div className="p-2 rounded-lg bg-[var(--highlight)] text-[var(--accent)]">
                        <FaBrain size={18} />
                      </div>
                      <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-[9px] font-bold uppercase border ${
                        info.available 
                        ? 'bg-green-100/10 border-green-500/20 text-green-500' 
                        : 'bg-red-100/10 border-red-500/20 text-red-500'
                      }`}>
                        {info.available ? <><FaCheckCircle /> Ready</> : <><FaTimesCircle /> Missing Weights</>}
                      </div>
                    </div>

                    <div className="space-y-1">
                      <h3 className="text-sm font-bold tracking-tight">{info.label}</h3>
                      <p className="text-[10px] text-gray-500 font-medium uppercase tracking-tighter">ID: {id}</p>
                    </div>

                    <div className="mt-6 pt-4 border-t border-[var(--border)] grid grid-cols-2 gap-4">
                      <div className="space-y-0.5">
                        <span className="text-[9px] font-bold uppercase text-gray-400">Input Size</span>
                        <p className="text-xs font-mono font-semibold">{info.input_size}</p>
                      </div>
                      <div className="space-y-0.5">
                        <span className="text-[9px] font-bold uppercase text-gray-400">Class Out</span>
                        <p className="text-xs font-mono font-semibold">{info.num_classes}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </LoadingOverlay>
          </section>

          {/* Local Priority Info Card */}
          <div className="card border-dashed bg-[var(--highlight)] flex items-center gap-4 p-6">
            <div className="w-10 h-10 rounded-full bg-[var(--accent)]/10 flex items-center justify-center text-[var(--accent)]">
              <FaMicrochip />
            </div>
            <div className="space-y-0.5">
              <p className="text-xs font-bold">Local-First Architecture</p>
              <p className="text-[10px] text-gray-500">The system is configured to prioritize local `.pth` and `.pkl` assets. No cloud sync is required for inference.</p>
            </div>
          </div>
        </div>
      </main>

      <Footer />
      <ToastContainer position="top-right" autoClose={3000} hideProgressBar={false} />
    </div>
  );
};

export default ModelPage;
