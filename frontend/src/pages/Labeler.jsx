import React, { useEffect, useState } from 'react';
import * as XLSX from 'xlsx';
import ECGPlot from '../components/ECGPlot';
import LabelPanel from '../components/LabelPanel';

const Labeler = () => {
  const [sheetNames, setSheetNames] = useState([]);
  const [sheetData, setSheetData] = useState([]);
  const [selectedSheet, setSelectedSheet] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0); // 10 at a time

  useEffect(() => {
    const workbookData = JSON.parse(localStorage.getItem('excelData'));
    const names = JSON.parse(localStorage.getItem('excelSheets'));
    setSheetNames(names);
    if (names.length > 0) {
      setSelectedSheet(names[0]); // default
      const sheet = workbookData[names[0]];
      const json = XLSX.utils.sheet_to_json(sheet);
      setSheetData(json);
    }
  }, []);

  const handleSheetChange = (e) => {
    const name = e.target.value;
    const workbookData = JSON.parse(localStorage.getItem('excelData'));
    const sheet = workbookData[name];
    const json = XLSX.utils.sheet_to_json(sheet);
    setSelectedSheet(name);
    setSheetData(json);
    setCurrentIndex(0);
  };

  const showNext = () => {
    setCurrentIndex((prev) => prev + 10);
  };

  const showPrev = () => {
    setCurrentIndex((prev) => Math.max(0, prev - 10));
  };

  const currentBatch = sheetData.slice(currentIndex, currentIndex + 10);

  return (
    <div style={{ padding: 20 }}>
      <h2>📊 Label ECG Rows</h2>

      <label>Select Sheet: </label>
      <select value={selectedSheet} onChange={handleSheetChange}>
        {sheetNames.map((name, i) => (
          <option key={i} value={name}>{name}</option>
        ))}
      </select>

      {currentBatch.map((row, i) => (
        <div key={i} style={{ border: '1px solid #ccc', marginTop: 10, padding: 10 }}>
          <ECGPlot row={row} index={currentIndex + i} />
          <LabelPanel rowIndex={currentIndex + i} />
        </div>
      ))}

      <div style={{ marginTop: 20 }}>
        <button onClick={showPrev} disabled={currentIndex === 0}>← Previous</button>
        <button onClick={showNext} disabled={currentIndex + 10 >= sheetData.length}>Next →</button>
      </div>
    </div>
  );
};

export default Labeler;
