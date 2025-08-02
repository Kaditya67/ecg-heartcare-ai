import React, { useState } from 'react';
import * as XLSX from 'xlsx';
import { useNavigate } from 'react-router-dom';

const FileUploader = () => {
  const [file, setFile] = useState(null);
  const navigate = useNavigate();

  const handleUpload = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
  };

  const handleRead = () => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const data = new Uint8Array(e.target.result);
      const workbook = XLSX.read(data, { type: 'array' });

      const sheets = workbook.SheetNames;
      localStorage.setItem('excelSheets', JSON.stringify(sheets));
      localStorage.setItem('excelData', JSON.stringify(workbook.Sheets));

      navigate('/label');
    };
    reader.readAsArrayBuffer(file);
  };

  return (
    <div>
      <input type="file" accept=".xlsx, .xls" onChange={handleUpload} />
      <br />
      {file && <button onClick={handleRead}>Next →</button>}
    </div>
  );
};

export default FileUploader;
