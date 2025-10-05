import React, { useState, useEffect, useCallback } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import Footer from '../components/Footer';

const ModelPage = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [targetFolder, setTargetFolder] = useState('api/models'); // default folder
  const [loading, setLoading] = useState(false);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await API.get('/drive-models/', { params: { target_folder: targetFolder } });
      setModels(resp.data.models);
    } catch (err) {
      toast.error('Failed to fetch models.');
    } finally {
      setLoading(false);
    }
  }, [targetFolder]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleDownload = async () => {
    if (!selectedModel) {
      toast.warn('Select a model first!');
      return;
    }
    setLoading(true);
    try {
      const resp = await API.post('/drive-models/', { filename: selectedModel, target_folder: targetFolder });
      toast.success(`${selectedModel} downloaded successfully to ${resp.data.folder}`);
      await fetchModels();
    } catch (err) {
      toast.error(err.response?.data?.error || 'Download failed, try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <DashboardNavbar />
      <main style={{ flexGrow: 1, padding: '1.5rem', maxWidth: 800, margin: 'auto', width: '100%' }}>
        <h2 className="text-2xl font-semibold mb-6 max-w-md mx-auto">Manage Models</h2>

        <div className="border-2 border-dashed border-green-500 rounded-md p-6 max-w-md mx-auto space-y-4">
          <label className="block font-medium mb-2">Target Folder:</label>
          <input
            type="text"
            value={targetFolder}
            className="border border-gray-300 rounded px-2 py-1 w-full focus:outline-none focus:ring-2 focus:ring-green-400"
            disabled
          />

          <label className="block font-medium mt-4 mb-2">Available Models:</label>
          {loading ? (
            <p className="text-gray-500">Loading models...</p>
          ) : models.length === 0 ? (
            <p className="text-gray-500 text-sm">No models found.</p>
          ) : (
            <select
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              className="border border-gray-300 rounded px-2 py-1 w-full focus:outline-none focus:ring-2 focus:ring-green-400"
            >
              <option value="">-- Choose a model --</option>
              {models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          )}

          <button
            onClick={handleDownload}
            disabled={loading || !selectedModel}
            className={`mt-4 px-4 py-2 rounded text-white ${
              loading || !selectedModel ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            {loading ? 'Downloading...' : 'Download Model'}
          </button>
        </div>
      </main>
      <Footer />

      {/* Toast container to show notifications */}
      <ToastContainer position="top-right" autoClose={3000} hideProgressBar={false} />
    </div>
  );
};

export default ModelPage;
