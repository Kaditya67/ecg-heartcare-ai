import React, { useState, useEffect, useCallback, useRef, useContext } from 'react';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import Footer from '../components/Footer';
import { ThemeContext } from '../components/context/ThemeContext';
import { FaFileUpload, FaFileCsv, FaFileExcel, FaTrashAlt, FaHistory, FaCloudUploadAlt } from 'react-icons/fa';

const MAX_FILE_SIZE = 1064 * 1024 * 1024; // 1GB

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
  const { theme } = useContext(ThemeContext);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [loadingFiles, setLoadingFiles] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);
  const [customName, setCustomName] = useState('');
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

  const handleChooseFile = e => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setCustomName(file?.name.replace(/\.[^/.]+$/, ""));
    setErrorMsg('');
  };

  const handleFileUpload = async () => {
    setErrorMsg('');
    setUploadProgress(0);

    if (!selectedFile) {
      setErrorMsg('Select a file first!');
      return;
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
      setSelectedFile(null);
      setCustomName('');
      setUploadProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = null; // 2. Reset file input value
      }
    } catch (error) {
      if (error.response && error.response.data) {
        setErrorMsg(error.response.data.error || 'Upload failed, please try again.');
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
                    onClick={handleFileUpload}
                    disabled={uploading}
                    className="w-full py-2 bg-[var(--accent)] hover:bg-blue-600 text-white rounded-lg text-xs font-bold uppercase tracking-widest transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {uploading ? <><div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div> Processing</> : <><FaFileUpload /> Launch Upload</>}
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
