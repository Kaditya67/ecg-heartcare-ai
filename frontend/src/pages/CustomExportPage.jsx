import React, { useEffect, useState, useContext } from 'react';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import Footer from '../components/Footer';
import { ThemeContext } from '../components/context/ThemeContext';
import { FaFileDownload, FaTable, FaUserFriends, FaTags, FaInfoCircle } from 'react-icons/fa';

const LoadingOverlay = ({ loading, children }) => (
  <div className="relative">
    {children}
    {loading && (
      <div className="absolute inset-0 bg-[var(--bg)]/50 backdrop-blur-[1px] z-50 flex items-center justify-center rounded-lg">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-4 border-[var(--accent)] border-t-transparent rounded-full animate-spin"></div>
          <span className="text-xs font-bold text-[var(--accent)] uppercase tracking-widest">Processing...</span>
        </div>
      </div>
    )}
  </div>
);

const CustomExportPage = () => {
  const { theme } = useContext(ThemeContext);
  const [files, setFiles] = useState([]); // {file_name, total_records, selected, numRecords}
  const [labels, setLabels] = useState([]); // {id, name, count, selected, numRecords}
  const [patients, setPatients] = useState([]); // {patient_id, record_count, selected, numRecords}

  const [loading, setLoading] = useState(false);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [error, setError] = useState('');
  const [exporting, setExporting] = useState(false);

  // Fetch all files on mount
  useEffect(() => {
    const fetchFiles = async () => {
      setLoading(true);
      try {
        const resp = await API.get('/files/summary/');
        console.log('Fetched files:', resp.data);
        setFiles(
          resp.data.map((f) => ({
            file_name: f.file_name,
            total_records: f.total_records,
            selected: false,
            numRecords: f.total_records,
          }))
        );
      } catch {
        setError('Failed to load file list.');
      } finally {
        setLoading(false);
      }
    };
    fetchFiles();
  }, []);

  // Fetch labels and patients filtered by selected files
  useEffect(() => {
    const selectedFileNames = files.filter((f) => f.selected).map((f) => f.file_name);
    if (selectedFileNames.length === 0) {
      setLabels([]);
      setPatients([]);
      return;
    }

    const fetchDetails = async () => {
      setLoadingDetails(true);
      try {
        const resp = await API.post('/ecgrecords/labels-patients-by-files/', {
          files: selectedFileNames,
        });
        setLabels(
          resp.data.labels.map((l) => ({
            id: l.id,
            name: l.name,
            count: l.count,
            selected: true,
            numRecords: l.count,
          }))
        );
        setPatients(
          resp.data.patients.map((p) => ({
            patient_id: p.patient_id,
            record_count: p.record_count,
            selected: true,
            numRecords: p.record_count,
          }))
        );
      } catch {
        setError('Failed to load label and patient data.');
        setLabels([]);
        setPatients([]);
      } finally {
        setLoadingDetails(false);
      }
    };
    fetchDetails();
  }, [files]);

  // Handlers for files
  const toggleFileSelect = (fileName) => {
    setFiles(files.map((f) => (f.file_name === fileName ? { ...f, selected: !f.selected } : f)));
  };
  const updateFileNumRecords = (fileName, value) => {
    const file = files.find((f) => f.file_name === fileName);
    if (!file) return;
    const safeValue = Math.min(Math.max(value, 1), file.total_records);
    setFiles(files.map((f) => (f.file_name === fileName ? { ...f, numRecords: safeValue } : f)));
  };

  // Handlers for labels
  const toggleLabelSelect = (id) => {
    setLabels(labels.map((l) => (l.id === id ? { ...l, selected: !l.selected } : l)));
  };
  const updateLabelNumRecords = (id, value) => {
    const label = labels.find((l) => l.id === id);
    if (!label) return;
    const safeValue = Math.min(Math.max(value, 1), label.count);
    setLabels(labels.map((l) => (l.id === id ? { ...l, numRecords: safeValue } : l)));
  };

  // Handlers for patients
  const togglePatientSelect = (patientId) => {
    setPatients(patients.map((p) => (p.patient_id === patientId ? { ...p, selected: !p.selected } : p)));
  };
  const updatePatientNumRecords = (patientId, value) => {
    const patient = patients.find((p) => p.patient_id === patientId);
    if (!patient) return;
    const safeValue = Math.min(Math.max(value, 1), patient.record_count);
    setPatients(patients.map((p) => (p.patient_id === patientId ? { ...p, numRecords: safeValue } : p)));
  };

  // Calculate total records selected from labels (can extend to patients/files if needed)
  const totalSelectedRecords = labels.reduce((sum, l) => (l.selected ? sum + l.numRecords : sum), 0);

  // Export handler
  const handleExport = async (format) => {
    setExporting(true);
    try {
      const labelFilter = labels
        .filter((l) => l.selected)
        .map((l) => ({ id: l.id, numRecords: l.numRecords }));
      const patientFilter = patients
        .filter((p) => p.selected)
        .map((p) => ({ patient_id: p.patient_id, numRecords: p.numRecords }));
      const fileFilter = files
        .filter((f) => f.selected)
        .map((f) => ({ file_name: f.file_name, numRecords: f.numRecords }));

      const response = await API.post(
        '/ecgrecords/export/',
        {
          labels: labelFilter,
          patients: patientFilter,
          files: fileFilter,
          format,
        },
        { responseType: 'blob' }
      );


      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', format === 'xlsx' ? 'export.xlsx' : 'export.csv');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch {
      alert('Export failed');
    } finally {
      setExporting(false);
    }
  };

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--bg)] text-[var(--text)] font-semibold text-sm">
      Loading files...
    </div>
  );
  if (error) return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--bg)] text-red-500 font-semibold text-sm">
      {error}
    </div>
  );

  return (
    <div className="min-h-screen flex flex-col bg-[var(--bg)] text-[var(--text)]">
      <DashboardNavbar />
      
      <main className="flex-grow p-4 lg:p-10">
        <div className="max-w-[1000px] mx-auto space-y-8">
          
          <header className="flex justify-between items-end border-b border-[var(--border)] pb-6">
            <div className="space-y-1">
              <h1 className="text-2xl font-bold">Custom Data Export</h1>
              <p className="text-xs text-gray-500 font-medium uppercase tracking-wider">Filtered Dataset Generator</p>
            </div>
            <div className="text-right">
              <div className="text-[10px] uppercase font-bold text-gray-400 mb-1">Status</div>
              <div className="flex items-center gap-2 px-3 py-1 bg-[var(--highlight)] border border-[var(--border)] rounded-full text-[10px] font-bold">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                System Ready
              </div>
            </div>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* LEFT COLUMN: Input Selection */}
            <div className="space-y-6">
              <LoadingOverlay loading={loading}>
                <section className="card space-y-4">
                  <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--accent)] flex items-center gap-2">
                    <FaTable /> 1. Select Source Files
                  </h3>
                  <div className="space-y-2 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
                    {files.length === 0 ? (
                      <p className="text-[10px] text-gray-400 italic py-4">No files available.</p>
                    ) : (
                      files.map((file) => (
                        <div key={file.file_name} className="flex items-center justify-between p-2 rounded-lg bg-[var(--highlight)] border border-[var(--border)] group hover:border-[var(--accent)] transition-all">
                          <label className="flex items-center gap-3 cursor-pointer flex-grow">
                            <input
                              type="checkbox"
                              checked={file.selected}
                              onChange={() => toggleFileSelect(file.file_name)}
                              className="w-4 h-4 rounded border-[var(--border)] text-[var(--accent)] focus:ring-[var(--accent)]"
                            />
                            <span className="text-xs font-semibold">{file.file_name}</span>
                            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-[var(--bg)] border border-[var(--border)] text-gray-500 font-bold">{file.total_records} rec</span>
                          </label>
                          {file.selected && (
                            <div className="flex items-center gap-2">
                              <span className="text-[9px] text-gray-400 font-bold uppercase">Qty:</span>
                              <input
                                type="number"
                                min={1}
                                max={file.total_records}
                                value={file.numRecords}
                                onChange={(e) => updateFileNumRecords(file.file_name, Number(e.target.value))}
                                className="w-16 bg-[var(--card-bg)] border border-[var(--border)] rounded px-1.5 py-0.5 text-[10px] font-bold outline-none focus:border-[var(--accent)]"
                              />
                            </div>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </section>
              </LoadingOverlay>

              <LoadingOverlay loading={loadingDetails}>
                <section className="card space-y-4">
                  <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--accent)] flex items-center gap-2">
                    <FaTags /> 2. Label Constraints
                  </h3>
                  <div className="space-y-2 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
                    {labels.length === 0 ? (
                      <p className="text-[10px] text-gray-400 italic py-4">Select files to see labels.</p>
                    ) : (
                      labels.map((label) => (
                        <div key={label.id} className="flex items-center justify-between p-2 rounded-lg bg-[var(--highlight)] border border-[var(--border)] hover:border-[var(--accent)] transition-all">
                          <label className="flex items-center gap-3 cursor-pointer flex-grow">
                            <input
                              type="checkbox"
                              checked={label.selected}
                              onChange={() => toggleLabelSelect(label.id)}
                              className="w-4 h-4"
                            />
                            <span className="text-xs font-semibold">{label.name}</span>
                            <span className="text-[9px] text-gray-400 font-bold">({label.count})</span>
                          </label>
                          {label.selected && (
                            <input
                              type="number"
                              min={1}
                              max={label.count}
                              value={label.numRecords}
                              onChange={(e) => updateLabelNumRecords(label.id, Number(e.target.value))}
                              className="w-16 bg-[var(--card-bg)] border border-[var(--border)] rounded px-1.5 py-0.5 text-[10px] font-bold"
                            />
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </section>
              </LoadingOverlay>
            </div>

            {/* RIGHT COLUMN: Patients and Summary */}
            <div className="space-y-6">
              <LoadingOverlay loading={loadingDetails}>
                <section className="card space-y-4">
                  <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--accent)] flex items-center gap-2">
                    <FaUserFriends /> 3. Patient Specifics
                  </h3>
                  <div className="space-y-2 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
                    {patients.length === 0 ? (
                      <p className="text-[10px] text-gray-400 italic py-4">Select files to see patients.</p>
                    ) : (
                      patients.map((patient) => (
                        <div key={patient.patient_id} className="flex items-center justify-between p-2 rounded-lg bg-[var(--highlight)] border border-[var(--border)] hover:border-[var(--accent)] transition-all">
                          <label className="flex items-center gap-3 cursor-pointer flex-grow">
                            <input
                              type="checkbox"
                              checked={patient.selected}
                              onChange={() => togglePatientSelect(patient.patient_id)}
                              className="w-4 h-4"
                            />
                            <span className="text-xs font-semibold">Patient {patient.patient_id}</span>
                          </label>
                          {patient.selected && (
                            <input
                              type="number"
                              min={1}
                              max={patient.record_count}
                              value={patient.numRecords}
                              onChange={(e) => updatePatientNumRecords(patient.patient_id, Number(e.target.value))}
                              className="w-16 bg-[var(--card-bg)] border border-[var(--border)] rounded px-1.5 py-0.5 text-[10px] font-bold"
                            />
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </section>
              </LoadingOverlay>

              <section className="card bg-[var(--highlight)] border-dashed border-2 flex flex-col items-center justify-center p-8 space-y-6">
                <div className="text-center space-y-2">
                  <h4 className="text-xs font-bold uppercase tracking-widest text-gray-400">Export Summary</h4>
                  <div className="text-4xl font-extrabold text-[var(--accent)]">{totalSelectedRecords}</div>
                  <p className="text-[10px] font-semibold text-gray-500 uppercase">Records Ready for Download</p>
                </div>

                <div className="flex w-full gap-3">
                  <button
                    disabled={exporting || totalSelectedRecords === 0}
                    onClick={() => handleExport('csv')}
                    className="flex-1 flex flex-col items-center justify-center gap-2 py-4 rounded-xl border border-[var(--border)] bg-[var(--card-bg)] text-[var(--text)] font-bold transition-all hover:bg-[var(--accent)] hover:text-white hover:border-[var(--accent)] disabled:opacity-30 disabled:grayscale"
                  >
                    <FaFileDownload size={20} />
                    <span className="text-[10px] uppercase tracking-widest">CSV Format</span>
                  </button>
                  <button
                    disabled={exporting || totalSelectedRecords === 0}
                    onClick={() => handleExport('xlsx')}
                    className="flex-1 flex flex-col items-center justify-center gap-2 py-4 rounded-xl border border-[var(--border)] bg-[var(--card-bg)] text-[var(--text)] font-bold transition-all hover:bg-green-600 hover:text-white hover:border-green-600 disabled:opacity-30 disabled:grayscale"
                  >
                    <FaTable size={20} />
                    <span className="text-[10px] uppercase tracking-widest">Excel Format</span>
                  </button>
                </div>

                <p className="text-[9px] text-gray-400 flex items-center gap-1">
                  <FaInfoCircle /> Exports include clinical metadata and labeled waveforms.
                </p>
              </section>
            </div>
          </div>
        </div>
      </main>

      <Footer />
      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: var(--accent); }
      `}} />
    </div>
  );
};

export default CustomExportPage;
