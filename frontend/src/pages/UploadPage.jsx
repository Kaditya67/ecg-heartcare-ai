import React, { useState, useEffect, useCallback, useRef, useContext } from 'react';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import Footer from '../components/Footer';
import { ThemeContext } from '../components/context/ThemeContext';
import { FaFileUpload, FaFileCsv, FaFileExcel, FaTrashAlt, FaHistory, FaCloudUploadAlt } from 'react-icons/fa';
import * as XLSX from 'xlsx';
import Papa from 'papaparse';

const MAX_FILE_SIZE = 1064 * 1024 * 1024; // 1GB
const REQUIRED_COLUMNS = ['patient_id', 'ecg_wave', 'heart_rate', 'label'];
const COLUMN_ALIASES = {
  patient_id: ['patient_id', 'Patient ID', 'patient id', 'PatientID', 'patientID', 'PatientId'],
  ecg_wave: ['ecg_wave', 'ECG Wave', 'ecg wave', 'ecgWave', 'ValueStr', 'ECG', 'Ecgwave'],
  heart_rate: ['heart_rate', 'Heart Rate', 'heart rate', 'Heartrate', 'Value', 'HeartRate', 'HR'],
  label: ['label', 'Label', 'Diagnosis', 'Class'],
};

const normalizeColumnKey = (value) => String(value ?? '').trim().toLowerCase().replace(/[\s-]+/g, '_');

const buildLocalMappingInfo = (columns = [], previewRows = []) => {
  const availableColumns = columns.map((column) => String(column));
  const normalizedAvailable = new Map(
    availableColumns.map((column) => [normalizeColumnKey(column), column])
  );

  const suggestedMapping = {};
  REQUIRED_COLUMNS.forEach((field) => {
    const candidates = [field, ...(COLUMN_ALIASES[field] || [])];
    const match = candidates
      .map((candidate) => normalizedAvailable.get(normalizeColumnKey(candidate)))
      .find(Boolean);
    if (match) {
      suggestedMapping[field] = match;
    }
  });

  return {
    available_columns: availableColumns,
    missing_required_columns: REQUIRED_COLUMNS.filter((field) => !suggestedMapping[field]),
    suggested_mapping: suggestedMapping,
    preview_rows: previewRows.slice(0, 5).map((row) => {
      const normalizedRow = {};
      availableColumns.forEach((column) => {
        const value = row?.[column];
        normalizedRow[column] = value == null ? '' : String(value);
      });
      return normalizedRow;
    }),
  };
};

const parseSelectedFile = (file) => new Promise((resolve, reject) => {
  const extension = file.name.split('.').pop()?.toLowerCase();

  if (extension === 'csv') {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      preview: 5,
      complete: (results) => {
        const columns = results.meta?.fields || [];
        resolve(buildLocalMappingInfo(columns, results.data || []));
      },
      error: (error) => reject(error),
    });
    return;
  }

  if (extension === 'xls' || extension === 'xlsx') {
    file.arrayBuffer()
      .then((data) => {
        const workbook = XLSX.read(data, { type: 'array' });
        const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
        const previewRows = XLSX.utils.sheet_to_json(firstSheet, { defval: '' }).slice(0, 5);
        const headerRow = XLSX.utils.sheet_to_json(firstSheet, { header: 1, defval: '' })[0] || [];
        resolve(buildLocalMappingInfo(headerRow, previewRows));
      })
      .catch(reject);
    return;
  }

  reject(new Error('Only CSV, XLSX, or XLS files are supported.'));
});

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

const UploadPage = () => {
  useContext(ThemeContext);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [parsingFile, setParsingFile] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);
  const [customName, setCustomName] = useState('');
  const [mappingInfo, setMappingInfo] = useState(null);
  const [columnMapping, setColumnMapping] = useState({});
  const fileInputRef = useRef(null); // 1. Create ref

  const fetchFiles = useCallback(async () => {
    setLoadingFiles(true);
    try {
      const resp = await API.get('/ecgfiles/');
      setFiles(resp.data.results);
    } catch (err) {
      setErrorMsg('Failed to load files.');
    } finally {
      setLoadingFiles(false);
    }
  }, []);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const handleChooseFile = async (e) => {
    const file = e.target.files[0];
    setSelectedFile(file || null);
    setCustomName(file?.name.replace(/\.[^/.]+$/, "") || '');
    setErrorMsg('');
    setMappingInfo(null);
    setColumnMapping({});

    if (!file) {
      return;
    }

    setParsingFile(true);
    try {
      const localMappingInfo = await parseSelectedFile(file);
      setMappingInfo(localMappingInfo);
      setColumnMapping(
        Object.fromEntries(
          REQUIRED_COLUMNS.map((field) => [field, localMappingInfo.suggested_mapping?.[field] || ''])
        )
      );
    } catch (error) {
      setErrorMsg(error.message || 'Could not read the selected file.');
    } finally {
      setParsingFile(false);
    }
  };

  const resetUploadState = () => {
    setSelectedFile(null);
    setCustomName('');
    setUploadProgress(0);
    setMappingInfo(null);
    setColumnMapping({});
    setParsingFile(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = null;
    }
  };

  const handleFileUpload = async (withMapping = false) => {
    setErrorMsg('');
    setUploadProgress(0);

    if (!selectedFile) {
      setErrorMsg('Select a file first!');
      return;
    }

    const shouldSendMapping = withMapping || Object.values(columnMapping).some(Boolean);

    if (shouldSendMapping && mappingInfo?.available_columns?.length) {
      const missingMappings = REQUIRED_COLUMNS.filter((col) => !columnMapping[col]);
      if (missingMappings.length > 0) {
        setErrorMsg(`Map all required fields before retrying: ${missingMappings.join(', ')}`);
        return;
      }
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setErrorMsg('File too large! Max size is 1GB.');
      return;
    }
    setUploading(true);

    const parts = selectedFile.name.split('.');
    const ext = parts.pop();
    const defaultName = parts.join('.');
    const newName = `${customName || defaultName}.${ext}`;

    const fileToUpload = new File([selectedFile], newName, { type: selectedFile.type });
    const formData = new FormData();
    formData.append('file', fileToUpload);
    if (shouldSendMapping) {
      const sanitizedMapping = Object.fromEntries(
        REQUIRED_COLUMNS.map((field) => [field, columnMapping[field] || ''])
      );
      formData.append('column_mapping', JSON.stringify(sanitizedMapping));
      REQUIRED_COLUMNS.forEach((field) => {
        formData.append(`column_mapping_${field}`, sanitizedMapping[field]);
      });
    }

    try {
      await API.post('/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            setUploadProgress(Math.round(100 * progressEvent.loaded / progressEvent.total));
          }
        }
      });
      await fetchFiles();
      resetUploadState();
    } catch (error) {
      if (error.response && error.response.data) {
        const payload = error.response.data;
        if (payload.available_columns && payload.missing_required_columns) {
          setMappingInfo(payload);
          setColumnMapping((prev) => ({
            ...prev,
            ...Object.fromEntries(
              REQUIRED_COLUMNS.map((col) => [col, payload.suggested_mapping?.[col] || prev[col] || ''])
            ),
          }));
          setErrorMsg(payload.error || 'Column mapping is required for this file.');
        } else {
          setErrorMsg(payload.error || 'Upload failed, please try again.');
        }
      } else {
        setErrorMsg('Upload failed, please try again.');
      }
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id) => {
    try {
      await API.delete(`/ecgfiles/${id}/`);
      await fetchFiles(); // await here too!
    } catch (error) {
      setErrorMsg('Delete failed, please try again.');
    }
  };

  const handleDownload = (url) => {
    if (url) {
      window.open(url, '_blank');
    } else {
      alert('Download URL not available.');
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[var(--bg)] text-[var(--text)] transition-colors duration-300">
      <DashboardNavbar />
      
      <main className="flex-grow p-4 lg:p-10">
        <div className="max-w-[800px] mx-auto space-y-10">
          
          <header className="flex flex-col items-center text-center space-y-2 border-b border-[var(--border)] pb-8">
            <h1 className="text-3xl font-bold tracking-tight">Dataset Management</h1>
            <p className="text-xs font-bold uppercase tracking-widest text-gray-500">Secure ECG File Ingestion & Storage</p>
          </header>

          {/* Upload Section */}
          <section className="card bg-[var(--highlight)] border-dashed border-2 p-8 flex flex-col items-center justify-center space-y-6">
            <div className="w-16 h-16 rounded-full bg-[var(--bg)] flex items-center justify-center text-[var(--accent)] text-3xl shadow-lg border border-[var(--border)]">
              <FaCloudUploadAlt />
            </div>
            
            <div className="text-center space-y-1">
              <h3 className="text-sm font-bold">Inbound Channel</h3>
              <p className="text-[10px] text-gray-500 uppercase font-bold tracking-tighter">CSV, XLSX, or XLS (Max 1GB)</p>
            </div>

            <div className="w-full max-w-sm space-y-4">
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleChooseFile}
                className="block w-full text-[11px] text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-[10px] file:font-bold file:uppercase file:bg-[var(--accent)] file:text-white hover:file:bg-blue-600 cursor-pointer"
                ref={fileInputRef}
              />

              {selectedFile && (
                <div className="p-4 bg-[var(--card-bg)] rounded-xl border border-[var(--border)] shadow-sm space-y-4">
                  <div className="space-y-1">
                    <label className="text-[10px] font-bold uppercase text-gray-500">Destination Name</label>
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={customName}
                        onChange={e => setCustomName(e.target.value)}
                        className="flex-grow bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs font-bold focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
                        placeholder="Filename..."
                      />
                      <span className="text-[10px] font-bold text-gray-400">.{selectedFile.name.split('.').pop()}</span>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => handleFileUpload(false)}
                    disabled={uploading || parsingFile}
                    className="w-full py-2 bg-[var(--accent)] hover:bg-blue-600 text-white rounded-lg text-xs font-bold uppercase tracking-widest transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {parsingFile ? (
                      <><div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div> Reading Columns</>
                    ) : uploading ? (
                      <><div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div> Processing</>
                    ) : (
                      <><FaFileUpload /> Launch Upload</>
                    )}
                  </button>
                </div>
              )}

              {uploading && (
                <div className="space-y-2">
                  <div className="w-full bg-[var(--bg)] rounded-full h-1.5 overflow-hidden border border-[var(--border)]">
                    <div
                      className="bg-[var(--accent)] h-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <div className="flex justify-between items-center text-[9px] font-bold text-gray-500 uppercase">
                    <span>Transmitting Data</span>
                    <span className="text-[var(--accent)]">{uploadProgress}%</span>
                  </div>
                </div>
              )}
            </div>
          </section>

          {errorMsg && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-500/50 rounded-lg flex items-center gap-3 text-red-600 dark:text-red-400 text-xs font-bold">
              <span>⚠️</span> {errorMsg}
            </div>
          )}

          {mappingInfo?.available_columns?.length > 0 && (
            <section className="card space-y-5">
              <div className="space-y-1">
                <h3 className="text-sm font-bold">Column Mapping</h3>
                <p className="text-[11px] text-gray-500">
                  Choose which file column should be used for each required field before upload. We prefill likely matches when we can, and you can change them here if your CSV or Excel headers use different names.
                </p>
              </div>

              {mappingInfo.missing_required_columns?.length === 0 ? (
                <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-500/30 rounded-lg text-[11px] text-emerald-700 dark:text-emerald-300">
                  All required fields were detected automatically. You can still adjust the mapping before uploading.
                </div>
              ) : (
                <div className="p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-500/30 rounded-lg text-[11px] text-amber-700 dark:text-amber-300">
                  Some required fields were not matched automatically. Please choose them below before uploading.
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {REQUIRED_COLUMNS.map((field) => (
                  <div key={field} className="space-y-1.5">
                    <label className="text-[10px] font-bold uppercase text-gray-500">
                      {field.replace('_', ' ')}
                    </label>
                    <select
                      value={columnMapping[field] || ''}
                      onChange={(e) => setColumnMapping((prev) => ({ ...prev, [field]: e.target.value }))}
                      className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs font-bold focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
                    >
                      <option value="">Select a column</option>
                      {mappingInfo.available_columns.map((column) => (
                        <option key={column} value={column}>
                          {column}
                        </option>
                      ))}
                    </select>
                    {mappingInfo.suggested_mapping?.[field] && (
                      <p className="text-[10px] text-gray-400">
                        Suggested: <span className="font-semibold">{mappingInfo.suggested_mapping[field]}</span>
                      </p>
                    )}
                  </div>
                ))}
              </div>

              {mappingInfo.preview_rows?.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-[10px] font-bold uppercase tracking-widest text-gray-500">Preview Rows</h4>
                  <div className="overflow-auto border border-[var(--border)] rounded-lg">
                    <table className="min-w-full text-left text-[11px]">
                      <thead className="bg-[var(--highlight)]">
                        <tr>
                          {mappingInfo.available_columns.map((column) => (
                            <th key={column} className="px-3 py-2 font-bold uppercase text-[10px] text-gray-500">
                              {column}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {mappingInfo.preview_rows.map((row, index) => (
                          <tr key={index} className="border-t border-[var(--border)]">
                            {mappingInfo.available_columns.map((column) => (
                              <td key={`${index}-${column}`} className="px-3 py-2 align-top text-[var(--text)]">
                                {row[column] || <span className="text-gray-400">-</span>}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="flex gap-3">
                <button
                  onClick={() => handleFileUpload(true)}
                  disabled={uploading || parsingFile}
                  className="px-4 py-2 bg-[var(--accent)] text-white rounded-lg text-xs font-bold uppercase tracking-widest hover:bg-blue-600 disabled:opacity-50"
                >
                  Upload With Mapping
                </button>
                <button
                  onClick={resetUploadState}
                  disabled={uploading || parsingFile}
                  className="px-4 py-2 border border-[var(--border)] rounded-lg text-xs font-bold uppercase tracking-widest hover:border-[var(--accent)]"
                >
                  Cancel
                </button>
              </div>
            </section>
          )}

          {/* History Section */}
          <section className="space-y-4">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-gray-500 px-2">
              <FaHistory /> File Repository
            </div>
            
            <LoadingOverlay loading={loadingFiles} text="Syncing Records...">
              <div className="space-y-3">
                {files.length === 0 && !loadingFiles ? (
                  <div className="p-10 text-center card border-dashed text-gray-400 italic text-xs">
                    No records found in current repository.
                  </div>
                ) : (
                  files.map((file) => (
                    <div
                      key={file.id}
                      className="card p-4 flex justify-between items-center group hover:border-[var(--accent)] transition-all"
                    >
                      <div className="space-y-1">
                        <p className="text-sm font-bold group-hover:text-[var(--accent)] transition-colors">{file.file_name}</p>
                        <p className="text-[10px] text-gray-500 font-bold uppercase tracking-tight">
                          <span className="text-[var(--accent)]">{file.record_count ?? 0}</span> Records • {new Date(file.uploaded_at).toLocaleDateString()}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <button onClick={() => handleDownload(file.download_csv_url)} className="p-2 bg-[var(--highlight)] text-gray-500 hover:text-[var(--accent)] hover:bg-[var(--bg)] border border-[var(--border)] rounded-lg transition-all" title="Download CSV">
                          <FaFileCsv size={14} />
                        </button>
                        <button onClick={() => handleDownload(file.download_xlsx_url)} className="p-2 bg-[var(--highlight)] text-gray-500 hover:text-green-500 hover:bg-[var(--bg)] border border-[var(--border)] rounded-lg transition-all" title="Download Excel">
                          <FaFileExcel size={14} />
                        </button>
                        <button
                          onClick={() => handleDelete(file.id)}
                          className="p-2 bg-[var(--highlight)] text-gray-500 hover:text-red-500 hover:bg-[var(--bg)] border border-[var(--border)] rounded-lg transition-all ml-2"
                          title="Purge Record"
                        >
                          <FaTrashAlt size={14} />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </LoadingOverlay>
          </section>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default UploadPage;
