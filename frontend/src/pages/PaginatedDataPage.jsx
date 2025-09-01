  import React, { useEffect, useState, useContext } from 'react';
  import API from '../api/api';
  import ECGCharts from '../components/ECGCharts';
  import DashboardNavbar from '../components/DashboardNavbar';
  import { ThemeContext } from '../components/context/ThemeContext';
  import Footer from '../components/Footer';

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

  const PaginatedDataPage = () => {
    const { theme } = useContext(ThemeContext);
    const [readyTheme, setReadyTheme] = useState(theme);

    // Features
    const [autoMove, setAutoMove] = useState(false);
    const [showDescription, setShowDescription] = useState(true);


    // File and data states
    const [files, setFiles] = useState([]);
    const [data, setData] = useState([]);
    const [plotRow, setPlotRow] = useState(null);
    const [labelOptions, setLabelOptions] = useState([]);
    const [chartType, setChartType] = useState('plotly');

    // Pagination states
    const [page, setPage] = useState(1);
    const [inputValue, setInputValue] = useState('1');
    const [pageCount, setPageCount] = useState(1);

    // UI control states
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [loadingPlotId, setLoadingPlotId] = useState(null);

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
      if (data.length === 0) return; // Don't re-fetch if no data has been loaded yet after mount/filter
      // Only fetch when user changes page within selected files (not file set), 
      // so store selected files at last filter apply
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
    const fetchData = async (pageNum = 1, selectedFileNames = null) => {
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
        };
        const response = await API.get('/records/', { params });
        const { results, count } = response.data;
        setData(results);
        setPageCount(Math.max(1, Math.ceil(count / PAGE_SIZE)));
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

    const handlePageInputChange = (e) => setInputValue(e.target.value);

    const fetchEcgData = async (id, patientId) => {
      setLoadingPlotId(id);
      setError('');
      try {
        const response = await API.get(`/records/${id}/wave/`, { params: { patient_id: patientId } });
        setPlotRow(response.data);
        if (response.data.label_options) setLabelOptions(response.data.label_options);
        console.log(response.data.label_options)
      } catch (err) {
        setError(err.response?.data?.detail || err.message || 'Error loading plot data');
        setPlotRow(null);
        setLabelOptions([]);
      } finally {
        setLoadingPlotId(null);
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
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <DashboardNavbar />
        <main style={{ flexGrow: 1, padding: '1.5rem', margin: 'auto', width: '100%' }}>
          <div
            className="max-w-6xl mx-auto p-6 space-y-6"
            style={{ backgroundColor: 'var(--bg)', color: 'var(--text)', minHeight: '100vh' }}
          >
            <h2 className="text-2xl font-semibold">Filter and Label ECG Records</h2>

            <section style={{ marginBottom: 16 }}>
              {files.length === 0 ? (
                <p>Loading files…</p>
              ) : (
                files.map(file => (
                  <label key={file.file_name} style={{ display: 'block', marginBottom: 4 }}>
                    <input
                      type="checkbox"
                      checked={file.selected}
                      onChange={() => handleFileToggle(file.file_name)}
                    />{' '}
                    {file.file_name} ({file.total_records})
                  </label>
                ))
              )}
            </section>

            {/* Change controls */}
            <div className="flex space-x-4 mb-6">
              <button
                onClick={applyFileFilter}
                disabled={files.filter(f => f.selected).length === 0}
                className="px-4 py-2 rounded text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Apply File Filter
              </button>

              <button
                onClick={handleSaveLabels}
                disabled={!hasUnsavedChanges}
                className={`px-4 py-2 rounded text-white ${
                  hasUnsavedChanges ? 'bg-[var(--accent)] hover:bg-blue-700' : 'bg-gray-400 cursor-not-allowed'
                }`}
              >
                {hasUnsavedChanges ? '(Unsaved Changes)' : 'Saved'}
              </button>

              <button
                onClick={handleClearChanges}
                disabled={!hasUnsavedChanges}
                className={`px-4 py-2 rounded text-white ${
                  hasUnsavedChanges ? 'bg-red-500 hover:bg-red-600' : 'bg-gray-400 cursor-not-allowed'
                }`}
              >
                Clear Changes
              </button>
            </div>
            {error && <p className="text-red-600 dark:text-red-400">{error}</p>}

            {plotRow && ecgArray.length > 0 && (
              <div className="bg-[var(--card-bg)] shadow rounded-lg border border-[var(--border)] p-6 space-y-6">
                {/* Plot header */}
                <div className="flex justify-between items-center">
                  <div>
                    <div className="flex items-center gap-8">
                      <span className="text-md font-semibold">Record ID: {plotRow.id}</span>
                      <span className="text-lg font-semibold">Patient ID: {plotRow.patient_id}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => navigatePlot('prev')}
                      disabled={data.findIndex(r => r.id === plotRow.id) === 0}
                      className="px-3 py-1 bg-[var(--secondary)] rounded hover:bg-[#c2e2ff] disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      ← Prev
                    </button>
                    <button
                      onClick={() => navigatePlot('next')}
                      disabled={data.findIndex(r => r.id === plotRow.id) === data.length - 1}
                      className="px-3 py-1 bg-[var(--secondary)] rounded hover:bg-[#c2e2ff] disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next →
                    </button>
                    <button
                      onClick={() => {
                        setPlotRow(null);
                        setLabelOptions([]);
                      }}
                      className="ml-4 text-[var(--danger)] hover:underline"
                    >
                      Close
                    </button>
                  </div>
                </div>

                {/* Chart and label controls */}
                <div className="flex flex-col md:flex-row gap-8 items-start">
                  <div className="flex-1 flex flex-col items-center">
                    <div className="mb-3 self-end">
                      <label className="mr-2">Chart Type:</label>
                      <select
                        value={chartType}
                        onChange={e => setChartType(e.target.value)}
                        className="border rounded p-1 bg-[var(--card-bg)] text-[var(--text)]"
                      >
                        <option value="plotly">Plotly</option>
                        <option value="chartjs">Chart.js</option>
                        <option value="echarts">ECharts</option>
                      </select>
                    </div>
                    <div className="w-full" style={{ minHeight: '400px', maxWidth: 900 }}>
                      <ECGCharts
                        ecgArray={ecgArray}
                        chartType={chartType}
                        theme={readyTheme}
                        key={readyTheme + chartType}
                      />
                    </div>
                  </div>

                  <div className="w-full md:w-64 border-l border-[var(--border)] pl-4 space-y-5">
                    {labelOptions.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {labelOptions.map(opt => {
                          const currentLabel = getCurrentLabel(plotRow.patient_id, plotRow.id, plotRow.label);
                          const isSelected = currentLabel === opt.value;
                          return (
                            <div className="relative group inline-block">
                              <button
                                key={opt.value}
                                className={`px-3 py-1 rounded text-xs font-medium ${
                                  isSelected ? 'text-white' : 'text-[var(--text)] border'
                                }`}
                                style={{ backgroundColor: isSelected ? opt.color : 'transparent' }}
                                onClick={() =>
                                  handleLabelButtonClick(plotRow.patient_id, plotRow.id, opt.value)
                                }
                              >
                                {opt.name}
                              </button>

                              {/* Tooltip */}
                              {showDescription && (
                              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 
                                              hidden group-hover:flex flex-col items-center z-20">
                                {/* Bubble */}
                                <div className="w-64 rounded-lg bg-black text-white text-sm px-3 py-2 
                                                leading-snug shadow-lg opacity-0 group-hover:opacity-100 
                                                transition duration-300 ease-out">
                                  {opt.description}
                                </div>
                                {/* Arrow */}
                                <div className="w-3 h-3 bg-black rotate-45 mt-[-6px]"></div>
                              </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                    <div className="text-sm text-[var(--text)] space-y-1">
                      <p>
                        <strong>Heart Rate:</strong> {plotRow.heart_rate || 'Unknown'}
                      </p>
                      <p>
                        <strong>Label:</strong> {getCurrentLabel(plotRow.patient_id, plotRow.id, plotRow.label)}
                      </p>
                      <p>
                        <strong>Source:</strong> {plotRow.source || 'Unknown'}
                      </p>
                      <p>
                        <strong>Total Points:</strong> {ecgArray.length}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Settings panel */}
            <div className="flex items-center gap-4 mb-4">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={autoMove}
                  onChange={() => setAutoMove(prev => !prev)}
                />
                Auto move to next
              </label>

              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={showDescription}
                  onChange={() => setShowDescription(prev => !prev)}
                />
                Show descriptions
              </label>
            </div>

            {/* Data table & pagination */}
            {(!loading && !error && data.length>0) ? (
              <>
                <div
                  className="overflow-x-auto border border-[var(--border)] rounded bg-[var(--card-bg)] shadow"
                  style={{ maxHeight: '300px', overflowY: 'auto' }}
                >
                  <table className="min-w-full table-auto text-sm divide-y divide-gray-200">
                    <thead className="bg-[var(--secondary)] sticky top-0 z-10">
                      <tr>
                        <th className="px-4 py-3 text-left font-semibold text-[var(--text)]">Record</th>
                        <th className="px-4 py-3 text-left font-semibold text-[var(--text)]">Patient ID</th>
                        <th className="px-4 py-3 text-center font-semibold text-[var(--text)]">Heart Rate</th>
                        <th className="px-4 py-3 text-center font-semibold text-[var(--text)]">Label</th>
                        <th className="px-4 py-3 text-center font-semibold text-[var(--text)]">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {data.map(({ id, patient_id, heart_rate, label }, idx) => {
                        const recordNumber = (page - 1) * PAGE_SIZE + idx + 1;
                        const isChanged = changedLabels[patient_id]?.[id] !== undefined;
                        const displayLabel = getLabelName(patient_id, id, label);
                        return (
                          <tr
                            key={id}
                            className={`cursor-pointer hover:bg-[var(--highlight)] ${
                              idx % 2 === 0 ? 'bg-[var(--card-bg)]' : ''
                            }`}
                          >
                            <td className="px-4 py-3 font-mono text-[var(--text)]">{recordNumber}</td>
                            <td className="px-4 py-3 text-[var(--text)]">{patient_id}</td>
                            <td className="px-4 py-3 text-center text-[var(--text)]">{heart_rate ?? 'N/A'}</td>
                            <td
                              className={`px-4 py-3 text-center font-medium ${
                                isChanged ? 'text-green-600' : 'text-[var(--text)]'
                              }`}
                            >
                              {displayLabel}
                            </td>
                            <td className="px-4 py-3 text-center">
                              <button
                                onClick={() => fetchEcgData(id, patient_id)}
                                disabled={loadingPlotId !== null}
                                className={`px-3 py-1 rounded text-white transition ${
                                  loadingPlotId === id ? 'bg-gray-400 cursor-not-allowed' : 'bg-[var(--accent)] hover:bg-blue-700'
                                }`}
                                aria-label={`Plot ECG record ${id} for patient ${patient_id}`}
                              >
                                {loadingPlotId === id ? 'Loading...' : 'Plot'}
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="flex justify-center items-center gap-6 mt-6 select-none">
                  <button
                    onClick={() => setInputValue(String(Number(inputValue) - 1))}
                    disabled={page <= 1}
                    className="px-5 py-2 border rounded-md bg-[var(--card-bg)] hover:bg-[#e0f2fe] disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Previous
                  </button>

                  <div className="flex items-center space-x-2">
                    <label htmlFor="pageInput" className="font-semibold text-[var(--text)]">
                      Page
                    </label>
                    <input
                      id="pageInput"
                      type="number"
                      min="1"
                      max={pageCount}
                      value={inputValue}
                      onChange={handlePageInputChange}
                      className="w-20 text-center border border-[var(--border)] rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                      aria-label="Page number input"
                    />
                    <span className="text-[var(--text)]">of {pageCount}</span>
                  </div>

                  <button
                    onClick={() => setInputValue(String(Number(inputValue) + 1))}
                    disabled={page >= pageCount}
                    className="px-5 py-2 border rounded-md bg-[var(--card-bg)] hover:bg-[#e0f2fe] disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next
                  </button>
                </div>
              </>
            ):
            (
              <p className="text-gray-500 mt-4">{loading ? 'Loading data...' : 'Apply filter...'}</p>
            )}
          </div>
        </main>
        <Footer />
      </div>
    );
  };

  export default PaginatedDataPage;
