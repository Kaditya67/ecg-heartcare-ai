import React, { useState, useEffect, useCallback, useContext } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import Footer from '../components/Footer';
import { ThemeContext } from '../components/context/ThemeContext';
import { FaCheckCircle, FaTimesCircle, FaBrain, FaCogs, FaMicrochip } from 'react-icons/fa';
import { getStoredRole, getStoredUser } from '../utils/auth';

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
  const role = getStoredRole();
  const currentUser = getStoredUser();
  const [modelRegistry, setModelRegistry] = useState({});
  const [patientAssignment, setPatientAssignment] = useState(null);
  const [allModels, setAllModels] = useState([]);
  const [assignments, setAssignments] = useState([]);
  const [patients, setPatients] = useState([]);
  const [settings, setSettings] = useState({ show_missing_weights: true });
  const [loading, setLoading] = useState(false);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [savingAssignment, setSavingAssignment] = useState(false);
  const [uploadingModel, setUploadingModel] = useState(false);
  const [assignmentForm, setAssignmentForm] = useState({ patient_id: '', model_key: '' });
  const [uploadForm, setUploadForm] = useState({
    label: '',
    key: '',
    base_model_key: 'ECG1DCNN',
    input_size: 2604,
    num_classes: 4,
    file: null,
  });
  const builtinChoices = allModels.filter((model) => model.source_type === 'builtin');
  const filteredModels = Object.entries(modelRegistry).filter(([id, info]) => {
    if (settings.show_missing_weights === false && !info.available) return false;
    return true;
  });
  const hiddenModelCount = Object.entries(modelRegistry).filter(([, info]) => !info.available).length;

  const fetchModels = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await API.get('/model_list/');
      setModelRegistry(resp.data.models || {});
      setPatientAssignment(resp.data.patient_assignment || null);
      if (role === 'admin') {
        const [registryResp, patientsResp] = await Promise.all([
          API.get('/models/registry/'),
          API.get('/patients/count/'),
        ]);
        setAllModels(registryResp.data.models || []);
        setAssignments(registryResp.data.assignments || []);
        setPatients(patientsResp.data || []);
      }
    } catch (err) {
      toast.error('Failed to fetch local model registry.');
    } finally {
      setLoading(false);
    }
  }, [role]);

  const fetchSettings = useCallback(async () => {
    setSettingsLoading(true);
    try {
      const response = await API.get('/profile/settings/');
      setSettings(response.data || { show_missing_weights: true });
    } catch (err) {
      toast.error('Failed to load profile settings.');
    } finally {
      setSettingsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  const handleUploadModel = async () => {
    if (!uploadForm.file) {
      toast.error('Choose a model file first.');
      return;
    }
    setUploadingModel(true);
    try {
      const formData = new FormData();
      formData.append('file', uploadForm.file);
      formData.append('label', uploadForm.label);
      formData.append('key', uploadForm.key);
      formData.append('base_model_key', uploadForm.base_model_key);
      formData.append('input_size', uploadForm.input_size);
      formData.append('num_classes', uploadForm.num_classes);
      await API.post('/models/registry/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      toast.success('Model uploaded.');
      setUploadForm((prev) => ({ ...prev, label: '', key: '', file: null }));
      await fetchModels();
    } catch (err) {
      toast.error(err.response?.data?.error || 'Failed to upload model.');
    } finally {
      setUploadingModel(false);
    }
  };

  const handleAssignModel = async () => {
    if (!assignmentForm.patient_id || !assignmentForm.model_key) {
      toast.error('Select both patient and model.');
      return;
    }
    setSavingAssignment(true);
    try {
      await API.post('/models/assignments/', assignmentForm);
      toast.success('Patient model assignment saved.');
      await fetchModels();
    } catch (err) {
      toast.error(err.response?.data?.error || 'Failed to save assignment.');
    } finally {
      setSavingAssignment(false);
    }
  };

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
              {settings.show_missing_weights === false && hiddenModelCount > 0 && (
                <div className="rounded-lg border border-yellow-300/30 bg-yellow-50/70 p-4 text-[11px] text-yellow-700 mb-4">
                  Hidden {hiddenModelCount} model{hiddenModelCount === 1 ? '' : 's'} with missing weights per your profile setting.
                </div>
              )}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredModels.map(([id, info]) => (
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

          {role === 'patient' && (
            <div className="card border-dashed bg-[var(--highlight)] p-6">
              <p className="text-xs font-bold">Assigned Patient Model</p>
              <p className="text-[10px] text-gray-500 mt-1">
                {patientAssignment
                  ? `Patient ${patientAssignment.patient_id} is assigned to ${patientAssignment.model_label}.`
                  : `No model is assigned yet for patient ${currentUser?.profile?.patient_id || '-'}.`}
              </p>
            </div>
          )}

          {role === 'admin' && (
            <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="card p-6 space-y-4">
                <div>
                  <h3 className="text-sm font-bold">Upload Trained Model</h3>
                  <p className="text-[10px] text-gray-500 mt-1">
                    Upload a trained `.pth` or `.pkl` file and register it as a deployable model.
                  </p>
                </div>
                <input
                  type="text"
                  placeholder="Display label"
                  value={uploadForm.label}
                  onChange={(e) => setUploadForm((prev) => ({ ...prev, label: e.target.value }))}
                  className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs"
                />
                <input
                  type="text"
                  placeholder="Optional key"
                  value={uploadForm.key}
                  onChange={(e) => setUploadForm((prev) => ({ ...prev, key: e.target.value }))}
                  className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs"
                />
                <select
                  value={uploadForm.base_model_key}
                  onChange={(e) => setUploadForm((prev) => ({ ...prev, base_model_key: e.target.value }))}
                  className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs"
                >
                  {builtinChoices.map((model) => (
                    <option key={model.key} value={model.key}>{model.key}</option>
                  ))}
                </select>
                <div className="grid grid-cols-2 gap-3">
                  <input
                    type="number"
                    placeholder="Input size"
                    value={uploadForm.input_size}
                    onChange={(e) => setUploadForm((prev) => ({ ...prev, input_size: e.target.value }))}
                    className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs"
                  />
                  <input
                    type="number"
                    placeholder="Num classes"
                    value={uploadForm.num_classes}
                    onChange={(e) => setUploadForm((prev) => ({ ...prev, num_classes: e.target.value }))}
                    className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs"
                  />
                </div>
                <input
                  type="file"
                  accept=".pth,.pkl"
                  onChange={(e) => setUploadForm((prev) => ({ ...prev, file: e.target.files?.[0] || null }))}
                  className="block w-full text-[11px]"
                />
                <button
                  onClick={handleUploadModel}
                  disabled={uploadingModel}
                  className="px-4 py-2 bg-[var(--accent)] text-white rounded-lg text-xs font-bold uppercase tracking-widest disabled:opacity-50"
                >
                  {uploadingModel ? 'Uploading...' : 'Upload Model'}
                </button>
              </div>

              <div className="card p-6 space-y-4">
                <div>
                  <h3 className="text-sm font-bold">Assign One Model Per Patient</h3>
                  <p className="text-[10px] text-gray-500 mt-1">
                    Patient users will only see the model assigned here.
                  </p>
                </div>
                <select
                  value={assignmentForm.patient_id}
                  onChange={(e) => setAssignmentForm((prev) => ({ ...prev, patient_id: e.target.value }))}
                  className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs"
                >
                  <option value="">Select patient</option>
                  {patients.map((patient) => (
                    <option key={patient.patient_id} value={patient.patient_id}>
                      Patient {patient.patient_id}
                    </option>
                  ))}
                </select>
                <select
                  value={assignmentForm.model_key}
                  onChange={(e) => setAssignmentForm((prev) => ({ ...prev, model_key: e.target.value }))}
                  className="w-full bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs"
                >
                  <option value="">Select model</option>
                  {allModels.map((model) => (
                    <option key={model.key} value={model.key}>
                      {model.label} ({model.key})
                    </option>
                  ))}
                </select>
                <button
                  onClick={handleAssignModel}
                  disabled={savingAssignment}
                  className="px-4 py-2 bg-[var(--accent)] text-white rounded-lg text-xs font-bold uppercase tracking-widest disabled:opacity-50"
                >
                  {savingAssignment ? 'Saving...' : 'Save Assignment'}
                </button>

                <div className="pt-4 border-t border-[var(--border)] space-y-2">
                  <p className="text-[10px] font-bold uppercase text-gray-500">Current Assignments</p>
                  <div className="max-h-48 overflow-auto space-y-2">
                    {assignments.map((assignment) => (
                      <div key={assignment.id} className="p-2 rounded-lg bg-[var(--highlight)] border border-[var(--border)] text-xs">
                        Patient {assignment.patient_id} {"->"} {assignment.model.label}
                      </div>
                    ))}
                    {assignments.length === 0 && (
                      <p className="text-[10px] text-gray-400 italic">No patient-specific model assignments yet.</p>
                    )}
                  </div>
                </div>
              </div>
            </section>
          )}

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
