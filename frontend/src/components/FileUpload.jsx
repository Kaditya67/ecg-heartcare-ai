import React from 'react';
import * as XLSX from 'xlsx';
import Papa from 'papaparse';

const FileUpload = ({ onUpload }) => {
  const handleChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const ext = file.name.split('.').pop().toLowerCase();

    let preview = [];

    if (ext === 'csv') {
      Papa.parse(file, {
        header: true,
        complete: (results) => {
          preview = results.data.slice(0, 5); // first 5 rows
          onUpload(file, preview);
        },
      });
    } else if (ext === 'xlsx') {
      const data = await file.arrayBuffer();
      const workbook = XLSX.read(data, { type: 'array' });
      const sheet = workbook.Sheets[workbook.SheetNames[0]];
      const json = XLSX.utils.sheet_to_json(sheet);
      preview = json.slice(0, 5);
      onUpload(file, preview);
    } else {
      alert('Only .xlsx and .csv files are supported');
    }

    e.target.value = '';
  };

  return (
    <div className="border-2 border-dashed border-[var(--border)] p-6 rounded-lg text-center bg-[var(--highlight)]">
      <p className="mb-2 text-sm">Upload your ECG Excel/CSV file</p>
      <input
        type="file"
        accept=".xlsx,.csv"
        onChange={handleChange}
        className="block mx-auto text-sm"
      />
    </div>
  );
};

export default FileUpload;
