import React, { useEffect, useState, useCallback, useContext } from 'react';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import { ThemeContext } from '../components/context/ThemeContext';
import Footer from '../components/Footer';
import { FaChartBar, FaTable, FaTrophy, FaMicrochip, FaExclamationCircle, FaCheckCircle, FaSync } from 'react-icons/fa';

// Lazy import echarts to avoid build-time errors
let ReactECharts = null;
try {
  ReactECharts = require('echarts-for-react').default;
} catch (e) {
  console.warn('echarts-for-react not found. Heatmap disabled.');
}

const LoadingOverlay = ({ loading, children, text = "Computing Metrics..." }) => (
  <div className="relative">
    {children}
    {loading && (
      <div className="absolute inset-0 bg-[var(--bg)]/50 backdrop-blur-[1px] z-50 flex items-center justify-center rounded-lg">
        <div className="flex flex-col items-center gap-3 text-center">
          <div className="w-8 h-8 border-4 border-[var(--accent)] border-t-transparent rounded-full animate-spin"></div>
          <div>
            <span className="text-[10px] font-bold text-[var(--accent)] uppercase tracking-widest block">{text}</span>
            <span className="text-[8px] text-gray-400 uppercase tracking-tighter">This audit may take up to 20 seconds</span>
          </div>
        </div>
      </div>
    )}
  </div>
);

const AnalysisPage = () => {
  const { theme } = useContext(ThemeContext);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedModel, setSelectedModel] = useState(null);

  const fetchAnalysis = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const resp = await API.get('/model-analysis/');
      setData(resp.data);
      // Auto-select the best model
      if (resp.data && resp.data.results) {
        const validModels = Object.entries(resp.data.results).filter(([, r]) => !r.error);
        if (validModels.length > 0) {
          const best = validModels.sort((a, b) => b[1].accuracy - a[1].accuracy)[0][0];
          setSelectedModel(best);
        }
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed. Ensure you have labeled data and model weights are loaded.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAnalysis();
  }, [fetchAnalysis]);

  const getConfusionOptions = () => {
    if (!data || !selectedModel || !data.results[selectedModel] || data.results[selectedModel].error) return {};
    const model = data.results[selectedModel];
    if (!model.confusion_matrix) return {};
    const matrix = model.confusion_matrix;
    const names = data.label_names || [];

    const heatmapData = [];
    for (let r = 0; r < matrix.length; r++) {
      for (let c = 0; c < (matrix[r] || []).length; c++) {
        heatmapData.push([c, r, matrix[r][c]]);
      }
    }

    const maxVal = heatmapData.reduce((acc, d) => Math.max(acc, d[2]), 1);

    return {
      backgroundColor: 'transparent',
      tooltip: { position: 'top', formatter: ({ value }) => `Actual → Predicted<br/><b>${value[2]} records</b>` },
      grid: { height: '65%', top: '8%', left: '18%', right: '5%' },
      xAxis: {
        type: 'category',
        data: names,
        name: 'Predicted',
        nameLocation: 'middle',
        nameGap: 28,
        splitArea: { show: true },
        axisLabel: { color: '#9ca3af', fontSize: 10, rotate: 30 }
      },
      yAxis: {
        type: 'category',
        data: names,
        name: 'Actual',
        nameLocation: 'middle',
        nameGap: 55,
        splitArea: { show: true },
        axisLabel: { color: '#9ca3af', fontSize: 10 }
      },
      visualMap: {
        min: 0,
        max: maxVal,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '2%',
        textStyle: { color: '#9ca3af', fontSize: 9 },
        inRange: {
          color: theme === 'dark' ? ['#1f2937', '#3b82f6'] : ['#f1f5f9', '#3b82f6']
        }
      },
      series: [{
        name: 'Confusion Matrix',
        type: 'heatmap',
        data: heatmapData,
        label: { show: true, color: '#fff', fontSize: 11, fontWeight: 'bold' },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.4)' } }
      }]
    };
  };

  const selectedModelData = data?.results?.[selectedModel];

  return (
    <div className="min-h-screen flex flex-col bg-[var(--bg)] text-[var(--text)] transition-colors duration-300">
      <DashboardNavbar />
      
      <main className="flex-grow container mx-auto px-4 py-10 max-w-6xl">
        <header className="mb-10 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight mb-1">Model Performance Analysis</h1>
            <p className="text-xs font-bold uppercase tracking-widest text-gray-500">Comparative Analytics & Diagnostics</p>
          </div>
          <button 
            onClick={fetchAnalysis} 
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 border border-[var(--border)] rounded-lg text-[10px] font-bold uppercase tracking-wider hover:border-[var(--accent)] transition-all disabled:opacity-50"
          >
            <FaSync className={loading ? 'animate-spin' : ''} /> Re-Audit
          </button>
        </header>

        {error && (
          <div className="card border-red-500/20 bg-red-500/5 p-6 flex flex-col items-center gap-3 text-center mb-10">
            <FaExclamationCircle className="text-red-500 text-2xl" />
            <div>
              <p className="text-xs font-bold text-red-500 uppercase tracking-widest">Diagnostic Failure</p>
              <p className="text-[10px] text-gray-500 mt-1">{error}</p>
            </div>
          </div>
        )}

        <LoadingOverlay loading={loading}>
          {data && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              {/* Leaderboard Column */}
              <div className="lg:col-span-4 space-y-6">
                <section className="card p-6">
                  <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-gray-500 mb-6">
                    <FaTrophy /> Performance Leaderboard
                  </div>
                  
                  <div className="space-y-2.5">
                    {Object.entries(data.results)
                      .sort((a, b) => (b[1].accuracy ?? -1) - (a[1].accuracy ?? -1))
                      .map(([id, info], idx) => (
                        <div 
                          key={id} 
                          onClick={() => !info.error && setSelectedModel(id)}
                          className={`p-3.5 rounded-xl border transition-all flex items-center justify-between ${
                            info.error ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'
                          } ${
                            selectedModel === id && !info.error
                            ? 'bg-[var(--accent)] border-[var(--accent)] text-white shadow-lg' 
                            : 'bg-[var(--highlight)] border-[var(--border)] hover:border-[var(--accent)]'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <span className={`text-[10px] font-black w-5 h-5 flex items-center justify-center rounded-full border
                              ${selectedModel === id ? 'border-white/30 text-white' : 'border-gray-400/20 text-gray-400'}`}>
                              {idx + 1}
                            </span>
                            <div>
                              <p className="text-[11px] font-bold uppercase tracking-tight leading-tight">{info.label || id}</p>
                              <p className={`text-[9px] uppercase tracking-wider ${selectedModel === id ? 'text-white/60' : 'text-gray-400'}`}>
                                {info.error ? 'Weights Missing' : `Acc: ${(info.accuracy * 100).toFixed(1)}%`}
                              </p>
                            </div>
                          </div>
                          {selectedModel === id && <FaCheckCircle size={13} className="text-white" />}
                        </div>
                      ))}
                  </div>
                </section>

                <section className="card p-5 bg-[var(--highlight)] border-dashed">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-[var(--accent)]/10 text-[var(--accent)]">
                      <FaMicrochip size={16} />
                    </div>
                    <div>
                      <p className="text-[11px] font-bold uppercase">Benchmark Dataset</p>
                      <p className="text-[10px] text-gray-500 mt-0.5">{data.sample_size} annotated records</p>
                    </div>
                  </div>
                </section>
              </div>

              {/* Confusion Matrix Column */}
              <div className="lg:col-span-8 space-y-6">
                {selectedModelData && !selectedModelData.error ? (
                  <>
                    <section className="card p-8">
                      <div className="flex items-center justify-between mb-8">
                        <div>
                          <h2 className="text-xl font-bold tracking-tight">{selectedModelData.label}</h2>
                          <p className="text-[10px] font-bold uppercase tracking-widest text-gray-500 mt-0.5">Confusion Matrix</p>
                        </div>
                        <div className="text-right">
                          <p className="text-4xl font-black text-[var(--accent)] leading-none">
                            {(selectedModelData.accuracy * 100).toFixed(1)}%
                          </p>
                          <p className="text-[9px] font-bold text-gray-400 uppercase tracking-widest mt-1">Global Accuracy</p>
                        </div>
                      </div>

                      <div className="h-[420px] w-full">
                        {ReactECharts ? (
                          <ReactECharts 
                            option={getConfusionOptions()} 
                            style={{ height: '100%', width: '100%' }}
                            theme={theme === 'dark' ? 'dark' : ''}
                            notMerge={true}
                          />
                        ) : (
                          <div className="h-full flex items-center justify-center text-gray-400 text-xs">
                            ECharts not available. Run <code className="mx-1 font-mono">npm install echarts</code> to enable heatmaps.
                          </div>
                        )}
                      </div>
                    </section>

                    <div className="grid grid-cols-2 gap-5">
                      <div className="card p-5 flex items-center justify-between">
                        <div>
                          <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Correct Predictions</p>
                          <p className="text-2xl font-bold mt-1">{selectedModelData.correct}</p>
                        </div>
                        <div className="p-3 rounded-full bg-green-500/10 text-green-500">
                          <FaCheckCircle size={16} />
                        </div>
                      </div>
                      <div className="card p-5 flex items-center justify-between">
                        <div>
                          <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Error Rate</p>
                          <p className="text-2xl font-bold mt-1">{((1 - selectedModelData.accuracy) * 100).toFixed(1)}%</p>
                        </div>
                        <div className="p-3 rounded-full bg-red-500/10 text-red-500">
                          <FaExclamationCircle size={16} />
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="card h-full flex flex-col items-center justify-center py-24 opacity-30 text-center space-y-4">
                    <FaTable size={40} />
                    <p className="text-xs font-bold uppercase tracking-widest px-10">
                      Select a model from the leaderboard to view its confusion matrix
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </LoadingOverlay>
      </main>

      <Footer />
    </div>
  );
};

export default AnalysisPage;
