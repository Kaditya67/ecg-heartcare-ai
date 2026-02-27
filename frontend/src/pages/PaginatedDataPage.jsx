  import React, { useEffect, useState, useContext } from 'react';
  import API from '../api/api';
  import ECGCharts from '../components/ECGCharts';
  import DashboardNavbar from '../components/DashboardNavbar';
  import { ThemeContext } from '../components/context/ThemeContext';
  import Footer from '../components/Footer';
  import ChatbotWidget from './ChatbotWidget';
  import { 
  FaFilter, FaSearch, FaSync, FaSave, FaCheckCircle, 
  FaChevronLeft, FaChevronRight, FaRobot, FaTimes, 
  FaCog, FaWaveSquare, FaPlus, FaBrain, FaListAlt 
} from 'react-icons/fa';

  function useDebounce(value, delay) {
    const [debounced, setDebounced] = useState(value);
    useEffect(() => {
      const handler = setTimeout(() => setDebounced(value), delay);
      return () => clearTimeout(handler);
    }, [value, delay]);
    return debounced;
  }

  const PAGE_SIZE = 100;
  const LOCAL_STORAGE_KEY = 'ecg_label_changes';

  const LoadingOverlay = ({ loading, children }) => (
    <div className="relative">
      {children}
      {loading && (
        <div className="absolute inset-0 bg-[var(--bg)]/50 backdrop-blur-[1px] z-50 flex items-center justify-center rounded-lg">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-4 border-[var(--accent)] border-t-transparent rounded-full animate-spin"></div>
            <span className="text-xs font-bold text-[var(--accent)] uppercase tracking-widest">Loading...</span>
          </div>
        </div>
      )}
    </div>
  );

  const PaginatedDataPage = () => {
    const { theme } = useContext(ThemeContext);
    const [readyTheme, setReadyTheme] = useState(theme);

    // Features
    const [autoMove, setAutoMove] = useState(true);
    const [showDescription, setShowDescription] = useState(false);


    // File and data states
    const [files, setFiles] = useState([]);
    const [data, setData] = useState([]);
    const [plotRow, setPlotRow] = useState(null);
    const [labelOptions, setLabelOptions] = useState([]);
    const [chartType, setChartType] = useState('echarts');
    const [waveCache, setWaveCache] = useState({}); // { recordId: waveData }

    // Pagination states
    const [page, setPage] = useState(1);
    const [inputValue, setInputValue] = useState('1');
    const [pageCount, setPageCount] = useState(1);

    // UI control states
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [loadingPlotId, setLoadingPlotId] = useState(null);
    const [models, setModels] = useState({});
    const [selectedModels, setSelectedModels] = useState([]);
    const [evalMode, setEvalMode] = useState(false);
    const [prediction, setPrediction] = useState(null);

    // Auto-label state
    const [autoLabelModel, setAutoLabelModel] = useState('ECG1DCNN');
    const [autoLabelResult, setAutoLabelResult] = useState(null);
    const [autoLabelLoading, setAutoLabelLoading] = useState(false);
    const [showAutoLabel, setShowAutoLabel] = useState(false);

    // Multi-Model Consensus State
    const [multiModelResults, setMultiModelResults] = useState(null);
    const [multiModelLoading, setMultiModelLoading] = useState(false);

    const [showConflictsOnly, setShowConflictsOnly] = useState(false);
    const [useRedis, setUseRedis] = useState(false);
    const [showSettings, setShowSettings] = useState(false);

    // Track label changes
    const [changedLabels, setChangedLabels] = useState(() => {
      try {
        return JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY)) || {};
      } catch {
        return {};
      }
    });

    // Fetch files on mount, select only the first file by default
    useEffect(() => {
      API.get('/files/summary/')
        .then(resp => {
          const filesData = resp.data;
          const initialFiles = filesData.map((f, i) => ({
            ...f,
            selected: i === 0
          }));
          setFiles(initialFiles);
        })
        .catch(() => setError('Failed to load files.'));
    }, []);

    useEffect(() => {
      if (evalMode) {
        API.get("/model_list/")
          .then((resp) => {
            setModels(resp.data); // { modelName: {...}, ... }
            console.log("Models fetched:", resp.data);
          })
          .catch((error) => setError("Error fetching models: " + error.message));
      } else {
        setModels({}); // clear models when Eval Mode is off
        setSelectedModels(""); // reset selection
      }
    }, [evalMode]);


    // Sync theme variable asynchronously for style updates
    useEffect(() => {
      const timeout = setTimeout(() => setReadyTheme(theme), 0);
      return () => clearTimeout(timeout);
    }, [theme]);

    // Debounce for page number input
    const debouncedInputValue = useDebounce(inputValue, 800);

    // Update page state from debounced input
    useEffect(() => {
      let val = Number(debouncedInputValue);
      if (isNaN(val) || val < 1) val = 1;
      else if (val > pageCount) val = pageCount;

      if (val !== page) setPage(val);
    }, [debouncedInputValue, pageCount, page]);

    // FETCH DATA ONLY WHEN PAGE CHANGES AFTER FILTER APPLIED, NOT ON FILE CHANGES!
    useEffect(() => {
      // Only skip if no file filter has been applied yet (no files selected)
      if (!lastSelectedFilesRef.current || lastSelectedFilesRef.current.length === 0) return;
      fetchData(page, lastSelectedFilesRef.current);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [page]);
    

    // Store which files were selected on last apply so pagination uses them
    const [lastSelectedFiles, setLastSelectedFiles] = useState([]);
    const lastSelectedFilesRef = React.useRef([]);
    useEffect(() => {
      lastSelectedFilesRef.current = lastSelectedFiles;
    }, [lastSelectedFiles]);

    // Main fetchData function
    const fetchData = async (pageNum = 1, selectedFileNames = null, conflictOverride = null) => {
      setLoading(true);
      setError('');
      try {
        const filesToUse = selectedFileNames ?? lastSelectedFiles;
        if (!filesToUse || filesToUse.length === 0) {
          setData([]);
          setPageCount(1);
          setLoading(false);
          return;
        }
        const params = {
          page: pageNum,
          page_size: PAGE_SIZE,
          file__file_name__in: filesToUse.join(','),
          conflict_only: conflictOverride !== null ? (conflictOverride || undefined) : (showConflictsOnly || undefined),
          use_redis: useRedis,
        };
        const response = await API.get('/records/', { params });
        const { results, count } = response.data;
        setData(results);
        setPageCount(Math.max(1, Math.ceil(count / PAGE_SIZE)));

        // Aggressive Pre-fetch: Load first 10 waves of the page in background
        results.slice(0, 10).forEach(rec => {
          if (!waveCache[rec.id]) {
            API.get(`/records/${rec.id}/wave/`, { params: { use_redis: useRedis } }).then(res => {
              setWaveCache(prev => ({ ...prev, [rec.id]: res.data }));
            }).catch(() => {});
          }
        });

      } catch (err) {
        setError(err.response?.data?.detail || err.message || 'Error loading data');
        setPlotRow(null);
        setLabelOptions([]);
      } finally {
        setLoading(false);
      }
    };

    // Only update files state on checkbox change
    const handleFileToggle = (fileName) => {
      setFiles(files.map(f =>
        f.file_name === fileName
          ? { ...f, selected: !f.selected }
          : f
      ));
      // Don't fetch data here!
    };

    // Only fetch data and save chosen files on apply
    const applyFileFilter = () => {
      setPage(1);
      setInputValue('1');
      const selectedFileNames = files.filter(f => f.selected).map(f => f.file_name);
      setLastSelectedFiles(selectedFileNames); // Save for further pagination
      fetchData(1, selectedFileNames);
    };

    // Wave Pre-fetching logic: pre-load next 3 records in background
    useEffect(() => {
      // If redis is disabled or no plot, don't pre-fetch to save bandwidth
      if (!useRedis || !plotRow || data.length === 0) return;
      
      const currentIndex = data.findIndex(r => r.id === plotRow.id);
      if (currentIndex === -1) return;

      // Identify next 3 records
      const nextRecords = data.slice(currentIndex + 1, currentIndex + 4);
      nextRecords.forEach(rec => {
        // Trigger a silent fetch to get it into Redis/Memory cache
        // Pass use_redis=true since we already checked useRedis above
        API.get(`/records/${rec.id}/wave/`, { params: { use_redis: true } }).catch(() => {});
      });
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [plotRow?.id, useRedis]);
    
    const handlePageInputChange = (e) => setInputValue(e.target.value);

    // Helper to keep page and input string in sync when using Prev/Next buttons
    const changePage = (newPage) => {
      setPage(newPage);
      setInputValue(String(newPage));
    };

    const handleConfirmAI = async (recordId, patientId, aiLabelVal) => {
      // Set the human label to match AI prediction
      setChangedLabels(prev => {
        const updated = {
          ...prev,
          [patientId]: {
            ...(prev[patientId] || {}),
            [recordId]: aiLabelVal
          }
        };
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updated));
        return updated;
      });
      // Optionally also verify it immediately
      await handleVerify(recordId, false); 
    };

    const fetchEcgData = async (id, patientId) => {
      setLoadingPlotId(id);
      setError('');
      // Also fetch multi-model consensus
      fetchMultiModelConsensus(id);

      // Check cache first
      if (waveCache[id]) {
        const cachedData = waveCache[id];
        setPlotRow(cachedData);
        if (cachedData.label_options) setLabelOptions(cachedData.label_options);
        setLoadingPlotId(null); // Ensure loading state is cleared for cached data
        return;
      }

      try {
        const response = await API.get(`/records/${id}/wave/`, { 
          params: { 
            patient_id: patientId,
            use_redis: useRedis // Pass toggle to backend
          } 
        });
        const waveData = response.data;
        setPlotRow(waveData);
        setWaveCache(prev => ({ ...prev, [id]: waveData }));
        if (waveData.label_options) setLabelOptions(waveData.label_options);
      } catch (err) {
        setError(err.response?.data?.detail || err.message || 'Error loading plot data');
        setPlotRow(null);
        setLabelOptions([]);
      } finally {
        setLoadingPlotId(null);
      }
    };

    const fetchMultiModelConsensus = async (recordId) => {
      setMultiModelLoading(true);
      setMultiModelResults(null);
      try {
        const resp = await API.post('/model-compare/', { record_id: recordId });
        setMultiModelResults(resp.data.model_results);
      } catch (err) {
        console.error('Failed to fetch multi-model consensus:', err);
      } finally {
        setMultiModelLoading(false);
      }
    };

    const handleLabelButtonClick = (patientId, recordId, labelValue) => {
      setChangedLabels(prev => ({
        ...prev,
        [patientId]: { ...(prev[patientId] || {}), [recordId]: labelValue },
      }));

      if (autoMove) {
        navigatePlot('next');
      }
    };

    const handleSaveLabels = async () => {
      const payload = [];
      Object.entries(changedLabels).forEach(([patientId, records]) => {
        Object.entries(records).forEach(([recordId, label]) => {
          payload.push({ id: Number(recordId), patient_id: patientId, label });
        });
      });
      try {
        await API.post('/records/bulk-label/', { records: payload });
        setChangedLabels({});
        localStorage.removeItem(LOCAL_STORAGE_KEY);
        fetchData(page, lastSelectedFiles);
        if (plotRow) fetchEcgData(plotRow.id, plotRow.patient_id);
      } catch {
        alert('Failed to save labels. Try again.');
      }
    };

    const handleClearChanges = () => {
      setChangedLabels({});
      localStorage.removeItem(LOCAL_STORAGE_KEY);
    };

    const handleAutoLabel = async () => {
      const selectedFileNames = files.filter(f => f.selected).map(f => f.file_name);
      if (selectedFileNames.length === 0) {
        alert('Select at least one file first.');
        return;
      }
      if (!window.confirm(
        `AI will label all records (without AI label) in ${selectedFileNames.length} file(s) using ${autoLabelModel}.\nThe AI labels go into a SEPARATE column — your manual labels are never changed. Continue?`
      )) return;

      setAutoLabelLoading(true);
      setAutoLabelResult(null);
      try {
        const allResults = [];
        for (const fname of selectedFileNames) {
          const resp = await API.post('/auto-label/', {
            file_name: fname,
            model_name: autoLabelModel,
            overwrite: false,
          });
          allResults.push(resp.data);
        }
        setAutoLabelResult(allResults);
        fetchData(page, lastSelectedFiles);
      } catch (err) {
        setAutoLabelResult([{ error: err.response?.data?.error || err.message }]);
      } finally {
        setAutoLabelLoading(false);
      }
    };

    const handleVerify = async (recordId, currentVerified) => {
      try {
        await API.post('/verify-label/', {
          record_id: recordId,
          verified: !currentVerified,
        });
        // Optimistic update
        setData(prev => prev.map(r =>
          r.id === recordId ? { ...r, is_verified: !currentVerified } : r
        ));
      } catch (err) {
        alert('Failed to update verification: ' + (err.response?.data?.error || err.message));
      }
    };

    const navigatePlot = (direction) => {
      if (!plotRow) return;
      const idx = data.findIndex(r => r.id === plotRow.id);
      if (idx === -1) return;
      let newIndex = direction === 'next' ? idx + 1 : idx - 1;
      if (newIndex < 0) newIndex = 0;
      if (newIndex >= data.length) newIndex = data.length - 1;
      if (data[newIndex]) fetchEcgData(data[newIndex].id, data[newIndex].patient_id);
    };

    let ecgArray = [];
    if (plotRow && typeof plotRow.ecg_wave === 'string') {
      ecgArray = plotRow.ecg_wave.split(',').map(Number);
    }

    const hasUnsavedChanges = Object.keys(changedLabels).some(pid =>
      Object.keys(changedLabels[pid]).length > 0
    );

    const getLabelName = (patientId, recordId, rawLabel) => {
      if (changedLabels[patientId]?.[recordId] !== undefined) {
        return labelOptions.find(l => l.value === changedLabels[patientId][recordId])?.name || '--Not Labeled--';
      }
      if (rawLabel && typeof rawLabel === 'object' && rawLabel.name) {
        return rawLabel.name;
      }
      return rawLabel ?? '--Not Labeled--';
    };

    const getCurrentLabel = (patientId, recordId, defaultLabel) =>
      changedLabels[patientId]?.[recordId] ?? defaultLabel ?? '--Not Labeled--';

    return (
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--bg)' }}>
        <DashboardNavbar />
        
        <main className="flex-grow p-4 lg:p-10">
          <div className="max-w-[1600px] mx-auto flex flex-col lg:flex-row gap-10 items-start">
            
            {/* LEFT COLUMN: Controls & Data Table */}
            <div className="w-full lg:w-[40%] sticky top-8 space-y-6">
              <div className="card space-y-4">
                <div className="flex justify-between items-center border-b border-[var(--border)] pb-3">
                  <h2 className="text-lg font-bold text-[var(--text)]">ECG Dataset</h2>
                  <div className="flex items-center gap-4">
                    {/* Main Redis Toggle (Promoted for visibility) */}
                    <div className="flex items-center gap-2 px-2 py-1 bg-[var(--highlight)] border border-[var(--border)] rounded-md">
                      <input 
                        type="checkbox" 
                        id="main_redis_toggle" 
                        checked={useRedis}
                        onChange={(e) => setUseRedis(e.target.checked)}
                        className="w-3.5 h-3.5 cursor-pointer"
                      />
                      <label htmlFor="main_redis_toggle" className="text-[10px] font-bold uppercase tracking-wider text-[var(--text)] cursor-pointer select-none">
                        Cache {useRedis ? <span className="text-green-500">Active</span> : <span className="text-amber-500">Slow</span>}
                      </label>
                    </div>

                    <button 
                      onClick={() => setShowSettings(!showSettings)}
                      className={`p-2 rounded-lg transition-colors ${showSettings ? 'bg-[var(--accent)] text-white' : 'hover:bg-[var(--highlight)] text-gray-500'}`}
                      title="Interface Settings"
                    >
                      <FaCog />
                    </button>
                  </div>
                </div>

                {/* Settings Panel (Consolidated) */}
                {showSettings && (
                  <div className="p-4 bg-[var(--highlight)] rounded-lg border border-[var(--border)] space-y-4">
                    <div className="flex justify-between items-center">
                      <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500">Advanced Settings</h4>
                      <button onClick={() => setShowSettings(false)} className="text-gray-400 hover:text-gray-600"><FaTimes size={10} /></button>
                    </div>
                    
                    <div className="grid grid-cols-1">
                      {/* Conflict Explorer */}
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id="conflict_toggle"
                          checked={showConflictsOnly}
                          onChange={(e) => {
                            setShowConflictsOnly(e.target.checked);
                            if (lastSelectedFiles.length > 0) fetchData(1, lastSelectedFiles, e.target.checked);
                          }}
                          className="w-4 h-4"
                        />
                        <label htmlFor="conflict_toggle" className="text-xs font-semibold text-[var(--text)] cursor-pointer">Show Conflicts Only</label>
                      </div>
                    </div>

                    {/* AI Auto-Label settings moved here */}
                    <div className="pt-3 border-t border-[var(--border)] space-y-3">
                      <div className="flex justify-between items-center">
                        <p className="text-[10px] uppercase font-bold text-gray-400">AI Background Jobs</p>
                      </div>
                      <div className="flex gap-2">
                        <select
                          value={autoLabelModel}
                          onChange={e => setAutoLabelModel(e.target.value)}
                          className="flex-1 bg-[var(--card-bg)] border border-[var(--border)] rounded px-2 py-1 text-xs text-[var(--text)]"
                        >
                          {Object.entries(models).length > 0
                            ? Object.entries(models).map(([name, info]) => (
                                <option key={name} value={name} disabled={!info.available}>{info.label}</option>
                              ))
                            : <option value="ECG1DCNN">ECG1DCNN (Baseline)</option>
                          }
                        </select>
                        <button
                          onClick={handleAutoLabel}
                          disabled={autoLabelLoading}
                          className="px-3 py-1 bg-[var(--accent)] text-white text-[10px] font-bold rounded hover:brightness-110 disabled:opacity-50"
                        >
                          {autoLabelLoading ? 'Running...' : 'Run Auto-Label'}
                        </button>
                      </div>
                      {autoLabelResult && (
                        <div className="text-[9px] p-2 bg-blue-50 dark:bg-blue-900/10 rounded border border-blue-100 dark:border-blue-800 text-blue-700 dark:text-blue-300">
                          Job completed. Refresh data to see results.
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* File Filter section */}
                <section className="space-y-1.5">
                  <h3 className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Selection Scope</h3>
                  <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto p-1">
                    {files.length === 0 ? (
                      <div className="w-full flex justify-center py-2"><div className="w-4 h-4 border-2 border-gray-300 border-t-gray-500 rounded-full animate-spin" /></div>
                    ) : (
                      files.map(file => (
                        <label 
                          key={file.file_name} 
                          className={`flex items-center gap-2 px-2 py-1 rounded border text-[10px] font-semibold transition-all cursor-pointer
                            ${file.selected 
                              ? 'bg-[var(--accent)] border-[var(--accent)] text-white' 
                              : 'bg-[var(--card-bg)] border-[var(--border)] text-[var(--text)] hover:border-[var(--accent)]'}`}
                        >
                          <input type="checkbox" className="hidden" checked={file.selected} onChange={() => handleFileToggle(file.file_name)} />
                          {file.file_name}
                        </label>
                      ))
                    )}
                  </div>
                </section>

                <div className="flex justify-center pt-1">
                  <button
                    onClick={applyFileFilter}
                    disabled={loading || files.filter(f => f.selected).length === 0}
                    className="px-6 py-1.5 bg-[var(--accent)] text-white text-[10px] font-bold uppercase tracking-wider rounded-full hover:shadow-md transition-all flex items-center gap-2 disabled:opacity-30"
                  >
                    <FaFilter size={10} /> Apply Selection
                  </button>
                </div>
              </div>

              {/* Data Table Section */}
              <div className="card !p-0 overflow-hidden">
                <LoadingOverlay loading={loading}>
                  <div className="overflow-x-auto overflow-y-auto max-h-[600px] relative">
                    <table className="min-w-full text-left">
                      <thead className="bg-[var(--highlight)] text-[var(--text)] border-b border-[var(--border)] sticky top-0 z-20">
                        <tr>
                          <th className="px-4 py-3 text-[10px] font-bold uppercase tracking-widest">ID</th>
                          <th className="px-4 py-3 text-[10px] font-bold uppercase tracking-widest">Patient</th>
                          <th className="px-4 py-3 text-[10px] font-bold uppercase tracking-widest text-center">Label</th>
                          <th className="px-4 py-3 text-[10px] font-bold uppercase tracking-widest text-center">AI</th>
                          <th className="px-4 py-3 text-[10px] font-bold uppercase tracking-widest text-center">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[var(--border)]">
                        {data.map(({ id, patient_id, heart_rate, label, ai_label, is_verified }, idx) => {
                          const isChanged = changedLabels[patient_id]?.[id] !== undefined;
                          const displayLabel = getLabelName(patient_id, id, label);
                          const humanVal = changedLabels[patient_id]?.[id] ?? label?.value;
                          const hasConflict = ai_label && humanVal !== undefined && humanVal !== ai_label.value;
                          const isActive = plotRow?.id === id;

                          return (
                            <tr
                              key={id}
                              onClick={() => fetchEcgData(id, patient_id)}
                              className={`cursor-pointer transition-colors border-l-4 
                                ${isActive ? 'bg-[var(--secondary)] border-l-[var(--accent)] shadow-inner' : 'hover:bg-[var(--highlight)] border-l-transparent'}
                                ${hasConflict && !isActive ? 'bg-red-500/5' : ''}
                              `}
                            >
                              <td className="px-4 py-3 text-[11px] font-medium opacity-50 font-mono">{(page - 1) * PAGE_SIZE + idx + 1}</td>
                              <td className="px-4 py-3">
                                <span className="font-bold text-xs text-[var(--text)] block">{patient_id}</span>
                                <span className="text-[9px] opacity-40 block font-mono">RECORD {id}</span>
                              </td>
                              <td className="px-4 py-3 text-center">
                                <span className={`text-[11px] font-bold px-2 py-0.5 rounded-full ${isChanged ? 'text-green-600 bg-green-50 border border-green-200' : 'text-[var(--text)] opacity-70'}`}>
                                  {displayLabel || 'None'}
                                </span>
                              </td>
                              <td className="px-4 py-3 text-center text-[11px]">
                                {ai_label ? (
                                  <span className="font-semibold text-blue-600 dark:text-blue-400">{ai_label.name}</span>
                                ) : <span className="opacity-20">—</span>}
                              </td>
                              <td className="px-4 py-3 text-center">
                                {is_verified ? (
                                  <span className="text-[9px] font-bold text-green-600 bg-green-50 border border-green-200 px-1.5 py-0.5 rounded uppercase">Verified</span>
                                ) : (
                                  hasConflict ? (
                                    <span className="text-[9px] font-bold text-red-600 bg-red-50 border border-red-200 px-1.5 py-0.5 rounded uppercase">Conflict</span>
                                  ) : <span className="text-[9px] text-gray-400 uppercase">Pending</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </LoadingOverlay>

                {/* Pagination UI */}
                <div className="px-4 py-3 bg-[var(--highlight)] flex justify-between items-center border-t border-[var(--border)]">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Page {page} of {pageCount}</span>
                  <div className="flex gap-2">
                    <button 
                      onClick={() => changePage(Math.max(1, page - 1))}
                      disabled={page === 1}
                      className="px-3 py-1 border border-[var(--border)] rounded text-[10px] font-bold uppercase tracking-wider disabled:opacity-30 bg-[var(--card-bg)] text-[var(--text)] hover:border-[var(--accent)] transition-colors"
                    >Prev</button>
                    <button 
                      onClick={() => changePage(Math.min(pageCount, page + 1))}
                      disabled={page === pageCount}
                      className="px-3 py-1 border border-[var(--border)] rounded text-[10px] font-bold uppercase tracking-wider disabled:opacity-30 bg-[var(--card-bg)] text-[var(--text)] hover:border-[var(--accent)] transition-colors"
                    >Next</button>
                  </div>
                </div>
              </div>
            </div>

            {/* RIGHT COLUMN: Sticky Signal Analyzer */}
            <div className="w-full lg:w-[60%] sticky top-8 space-y-6">
              <div className="card min-h-[600px] flex flex-col relative !p-6">
                <LoadingOverlay loading={loadingPlotId !== null}>
                  {plotRow ? (
                    <>
                      <div className="flex justify-between items-start mb-4 border-b border-[var(--border)] pb-4">
                        <div className="space-y-0.5">
                          <h3 className="text-xl font-bold text-[var(--text)]">Signal Analyzer</h3>
                          <p className="text-[11px] font-semibold text-gray-400">U-ID {plotRow.id} • Patient <span className="text-[var(--text)]">{plotRow.patient_id}</span> • Rate <span className="text-[var(--text)]">{plotRow.heart_rate ?? 'N/A'}</span></p>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="flex items-center gap-2 px-2 py-1 bg-[var(--highlight)] border border-[var(--border)] rounded-lg">
                            <span className="text-[9px] font-bold uppercase text-gray-400">Engine:</span>
                            <select 
                              value={chartType} 
                              onChange={e => setChartType(e.target.value)}
                              className="bg-transparent text-[var(--text)] text-[10px] font-bold outline-none cursor-pointer"
                            >
                              <option value="echarts">ECharts (Smooth)</option>
                              <option value="plotly">Plotly (Scientific)</option>
                              <option value="chartjs">Chart.js (Classic)</option>
                            </select>
                          </div>
                          <button
                            onClick={() => handleSaveLabels()}
                            disabled={!hasUnsavedChanges}
                            className={`px-4 py-1.5 rounded border font-bold text-[10px] uppercase tracking-widest transition-all
                              ${hasUnsavedChanges ? 'bg-green-600 border-green-600 text-white shadow-sm' : 'bg-[var(--highlight)] text-gray-400 border-[var(--border)]'}`}
                          >
                            <FaSave className="inline mr-1" /> Commit Changes
                          </button>
                        </div>
                      </div>

                    <div className="flex-grow bg-[var(--bg)] border border-[var(--border)] rounded-lg overflow-hidden relative shadow-inner">
                       <div className="h-full min-h-[400px]">
                          <ECGCharts
                            ecgArray={ecgArray}
                            chartType={chartType}
                            theme={readyTheme}
                            key={readyTheme + chartType + plotRow.id}
                          />
                       </div>
                    </div>

                    {/* Labeling Controls Overlay */}
                    <div className="mt-8 grid grid-cols-2 lg:grid-cols-4 gap-3">
                       {labelOptions.map(opt => {
                          const isSelected = getCurrentLabel(plotRow.patient_id, plotRow.id) === opt.value;
                          return (
                            <button
                              key={opt.value}
                              onClick={() => handleLabelButtonClick(plotRow.patient_id, plotRow.id, opt.value)}
                              className={`p-3 rounded-lg flex flex-col items-center gap-1 transition-colors border
                                ${isSelected 
                                  ? 'bg-[var(--accent)] border-[var(--accent)] text-white' 
                                  : 'bg-[var(--highlight)] border-[var(--border)] text-[var(--text)] hover:border-[var(--accent)]'}
                              `}
                            >
                              <span className="text-xs font-bold uppercase">{opt.name}</span>
                            </button>
                          );
                       })}
                     </div>

                     {/* Multi-Model Consensus Sidebar/Section */}
                     <div className="mt-8 pt-6 border-t border-[var(--border)] space-y-4">
                        <div className="flex items-center justify-between">
                           <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-gray-500">
                              <FaBrain className="text-[var(--accent)]" /> Model Consensus Leaderboard
                           </div>
                           {multiModelLoading && (
                             <div className="w-3 h-3 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin"></div>
                           )}
                        </div>

                        {!multiModelResults && !multiModelLoading ? (
                           <p className="text-[10px] text-gray-400 italic">Select a record to trigger consensus engine audit.</p>
                        ) : (
                           <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                              {multiModelResults && Object.entries(multiModelResults).map(([mId, res]) => (
                                 <div 
                                    key={mId}
                                    onClick={() => !res.error && plotRow && handleLabelButtonClick(plotRow.patient_id, plotRow.id, res.predicted_class)}
                                    className={`p-3 rounded-lg border transition-all cursor-pointer group flex items-center justify-between ${
                                      res.error 
                                      ? 'bg-red-50/5 border-red-500/10 opacity-50 grayscale' 
                                      : 'bg-[var(--highlight)] border-[var(--border)] hover:border-[var(--accent)]'
                                    }`}
                                 >
                                    <div className="space-y-0.5">
                                       <p className="text-[9px] font-bold uppercase text-gray-400 group-hover:text-[var(--accent)] transition-colors">{res.label || mId}</p>
                                       <p className={`text-xs font-bold ${res.error ? 'text-red-400' : 'text-[var(--text)]'}`}>
                                          {res.error ? 'Weights Missing' : res.predicted_class_name}
                                       </p>
                                    </div>
                                    {!res.error && (
                                       <div className="text-[9px] font-mono font-bold text-[var(--accent)] opacity-0 group-hover:opacity-100 transition-opacity">
                                          APPLY
                                       </div>
                                    )}
                                 </div>
                              ))}
                           </div>
                        )}
                     </div>

                      <div className="mt-8 flex justify-between items-center px-4 pt-4 border-t border-[var(--border)]">
                        <div className="flex gap-2">
                           <button onClick={() => navigatePlot('prev')} className="w-8 h-8 rounded border border-[var(--border)] flex items-center justify-center hover:bg-[var(--highlight)] text-[var(--text)] transition-all">←</button>
                           <button onClick={() => navigatePlot('next')} className="w-8 h-8 rounded border border-[var(--border)] flex items-center justify-center hover:bg-[var(--highlight)] text-[var(--text)] transition-all">→</button>
                        </div>
                        <div className="flex items-center gap-2">
                           <button
                             onClick={() => handleVerify(plotRow.id, !plotRow.is_verified)}
                             className={`px-4 py-1.5 rounded border font-bold text-[10px] uppercase tracking-wider transition-all
                               ${plotRow.is_verified 
                                ? 'bg-green-600 border-green-600 text-white shadow-sm' 
                                : 'bg-[var(--highlight)] border-[var(--border)] text-[var(--text)] hover:border-[var(--accent)]'}`}
                           >
                               {plotRow.is_verified ? <><FaCheckCircle className="inline mr-1" /> Verified</> : 'Verify'}
                           </button>
                        </div>
                    </div>
                  </>
                ) : (
                  <div className="flex-grow flex flex-col items-center justify-center space-y-3 py-20 text-center">
                    <div className="w-12 h-12 rounded-full bg-[var(--highlight)] flex items-center justify-center text-gray-300"><FaFilter size={24} /></div>
                    <div className="space-y-1">
                      <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest">Awaiting Selection</h3>
                      <p className="text-[10px] text-gray-400 font-medium">Select a patient record to begin analysis.</p>
                    </div>
                  </div>
                )}
                </LoadingOverlay>
              </div>
            </div>
          </div>
        </main>

        <ChatbotWidget />
        <Footer />

        <style dangerouslySetInnerHTML={{ __html: `
          body { overflow-x: hidden; }
        `}} />
      </div>
    );
  };

  export default PaginatedDataPage;
