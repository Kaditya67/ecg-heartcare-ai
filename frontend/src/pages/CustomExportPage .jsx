import React, { useEffect, useState } from 'react';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import Footer from '../components/Footer';

const CustomExportPage = () => {
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

  if (loading) return <div>Loading files...</div>;
  if (error) return <div className="text-red-600">{error}</div>;

  return (
    <>
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
        <h2 className="text-xl font-bold mb-4">Custom ECG Records Export</h2>

        <section className="mb-6">
          <h3 className="font-semibold mb-2">Select Files</h3>
          {files.length === 0 ? (
            <p>No files found.</p>
          ) : (
            files.map((file) => (
              <div key={file.file_name} className="mb-1 flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={file.selected}
                  onChange={() => toggleFileSelect(file.file_name)}
                />
                <span>
                  {file.file_name} ({file.total_records})
                </span>
                {file.selected && (
                  <input
                    type="number"
                    min={1}
                    max={file.total_records}
                    value={file.numRecords}
                    onChange={(e) => updateFileNumRecords(file.file_name, Number(e.target.value))}
                    className="ml-2 w-20 p-1 border rounded"
                  />
                )}
              </div>
            ))
          )}
        </section>

        {loadingDetails ? (
          <p>Loading labels and patients for selected files...</p>
        ) : (
          <>
            <section className="mb-6">
              <h3 className="font-semibold mb-2">Select Labels</h3>
              {labels.length === 0 ? (
                <p>No labels found for selected files.</p>
              ) : (
                labels.map((label) => (
                  <div key={label.id} className="mb-1 flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={label.selected}
                      onChange={() => toggleLabelSelect(label.id)}
                    />
                    <span>
                      {label.name} ({label.count})
                    </span>
                    {label.selected && (
                      <input
                        type="number"
                        min={1}
                        max={label.count}
                        value={label.numRecords}
                        onChange={(e) => updateLabelNumRecords(label.id, Number(e.target.value))}
                        className="ml-2 w-20 p-1 border rounded"
                      />
                    )}
                  </div>
                ))
              )}
            </section>

            <section className="mb-6">
              <h3 className="font-semibold mb-2">Select Patients</h3>
              {patients.length === 0 ? (
                <p>No patients found for selected files.</p>
              ) : (
                patients.map((patient) => (
                  <div key={patient.patient_id} className="mb-1 flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={patient.selected}
                      onChange={() => togglePatientSelect(patient.patient_id)}
                    />
                    <span>
                      {patient.patient_id} ({patient.record_count})
                    </span>
                    {patient.selected && (
                      <input
                        type="number"
                        min={1}
                        max={patient.record_count}
                        value={patient.numRecords}
                        onChange={(e) => updatePatientNumRecords(patient.patient_id, Number(e.target.value))}
                        className="ml-2 w-20 p-1 border rounded"
                      />
                    )}
                  </div>
                ))
              )}
            </section>
          </>
        )}

        <div className="mb-6 font-semibold">
          Total selected records (labels count): {totalSelectedRecords}
        </div>

        <div className="flex gap-4">
          <button
            disabled={exporting || totalSelectedRecords === 0}
            onClick={() => handleExport('csv')}
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
          >
            {exporting ? 'Exporting...' : 'Export CSV'}
          </button>
          <button
            disabled={exporting || totalSelectedRecords === 0}
            onClick={() => handleExport('xlsx')}
            className="px-4 py-2 bg-green-600 text-white rounded disabled:opacity-50"
          >
            {exporting ? 'Exporting...' : 'Export XLSX'}
          </button>
        </div>
      </main>
        <Footer />
      </div>
    </>
  );
};

export default CustomExportPage;
