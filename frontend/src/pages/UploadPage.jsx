import React, { useState, useEffect } from 'react';
import FileUpload from '../components/FileUpload';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';

const UploadPage = () => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [loadingFiles, setLoadingFiles] = useState(false);

  useEffect(() => {
    fetchFiles();
  }, []);

  // Fetch the list of uploaded files from the backend
  const fetchFiles = async () => {
    setLoadingFiles(true);
    try {
      const resp = await API.get('/files/'); // <-- Update to your files endpoint
      setFiles(resp.data);
    } catch (err) {
      setErrorMsg('Failed to load files.');
    } finally {
      setLoadingFiles(false);
    }
  };

  const handleFileUpload = async (file, preview) => {
    setErrorMsg('');
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      await API.post('/upload/', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      fetchFiles(); // Refresh the file list after upload
    } catch (error) {
      setErrorMsg('Upload failed, please try again.');
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  // New: Delete a file from the backend
  const handleDelete = async (id) => {
    try {
      await API.delete(`/files/${id}/`); // <-- Update to your file delete endpoint
      fetchFiles();
    } catch (error) {
      setErrorMsg('Delete failed, please try again.');
    }
  };

  // New: Download a file (assumes you have a download endpoint or file link)
  const handleDownload = (url) => {
    window.open(url, '_blank');
  };

  return (
    <>
      <DashboardNavbar />
      <div className="max-w-4xl mx-auto p-6">
        <h2 className="text-2xl font-semibold mb-6">Upload ECG Files</h2>
        <FileUpload onUpload={handleFileUpload} />
        {uploading && <p className="mt-4 text-sm text-blue-600">Uploading...</p>}
        {errorMsg && <p className="mt-4 text-sm text-red-600">{errorMsg}</p>}
        <div className="space-y-6 mt-8">
          {loadingFiles ? (
            <p className="text-gray-500">Loading files...</p>
          ) : files.length === 0 ? (
            <p className="text-sm text-gray-500">No files found.</p>
          ) : (
            files.map(file => (
              <div key={file.id} className="card space-y-3 p-4 border rounded-lg shadow-sm flex justify-between items-center">
                <div>
                  <p className="font-medium">{file.name}</p>
                  <p className="text-sm text-gray-500">{file.size} KB • uploaded at {file.uploaded_at}</p>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => handleDownload(file.download_url)}
                    className="button-normal"
                  >
                    Download
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
      </div>
    </>
  );
};

export default UploadPage;
