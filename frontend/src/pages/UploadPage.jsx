import React, { useState, useEffect, useCallback, useRef } from 'react';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import Footer from '../components/Footer';

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

const UploadPage = () => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [loadingFiles, setLoadingFiles] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);
  const [customName, setCustomName] = useState('');
  const fileInputRef = useRef(null);  // 1. Create ref

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
    if (!selectedFile) {
      setErrorMsg('Select a file first!');
      return;
    }
    if (selectedFile.size > MAX_FILE_SIZE) {
      setErrorMsg('File too large! Max size is 50MB.');
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
      await API.post('/upload/', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      await fetchFiles();
      setSelectedFile(null);
      setCustomName('');
      if (fileInputRef.current) {
        fileInputRef.current.value = null;    // 2. Reset file input value
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
      await fetchFiles();  // await here too!
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
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          minHeight: '100vh',
        }}
      >
        <DashboardNavbar />
        <main
          style={{
            flexGrow: 1,
            padding: '1.5rem',
            maxWidth: 800,
            margin: 'auto',
            width: '100%',
          }}
        >
        <h2 className="text-2xl font-semibold mb-6">Upload ECG Files</h2>

        <div className="border-2 border-dashed border-blue-500 rounded-md p-6 max-w-md mx-auto">
          <input
            type="file"
            accept=".csv,.xlsx"
            onChange={handleChooseFile}
            className="block w-full text-gray-700 cursor-pointer focus:outline-none"
            ref={fileInputRef}  
          />

          {selectedFile && (
            <div className="mt-4 flex items-center">
              <label className="mr-2 font-medium text-gray-700">Rename file:</label>
              <input
                type="text"
                value={customName}
                onChange={e => setCustomName(e.target.value)}
                className="border border-gray-300 rounded px-2 py-1 flex-grow focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
              <span className="ml-2 text-gray-500">.{selectedFile.name.split('.').pop()}</span>
              <button
                onClick={handleFileUpload}
                disabled={uploading}
                className={`ml-4 px-4 py-1 rounded text-white ${
                  uploading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {uploading ? 'Uploading...' : 'Upload'}
              </button>
            </div>
          )}
        </div>

        {errorMsg && <p className="mt-4 text-sm text-red-600">{errorMsg}</p>}

        <div className="space-y-6 mt-8">
          {loadingFiles ? (
            <p className="text-gray-500">Loading files...</p>
          ) : files.length === 0 ? (
            <p className="text-sm text-gray-500">No files found.</p>
          ) : (
            files.map((file) => (
              <div
                key={file.id}
                className="card space-y-3 p-4 border rounded-lg shadow-sm flex justify-between items-center"
              >
                <div>
                  <p className="font-medium">{file.file_name}</p>
                  <p className="text-sm text-gray-500">
                    {file.record_count ?? 0} records • uploaded at {new Date(file.uploaded_at).toLocaleString()}
                  </p>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => handleDownload(file.download_csv_url)}
                    className="button-normal"
                  >
                    CSV
                  </button>
                  <button
                    onClick={() => handleDownload(file.download_xlsx_url)}
                    className="button-normal"
                  >
                    XLS
                  </button>
                  <button
                    onClick={() => handleDelete(file.id)}
                    className="button-abnormal bg-red-500 text-white px-3 py-1 rounded hover:bg-red-600 transition"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default UploadPage;