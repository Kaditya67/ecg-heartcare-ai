import React, { useState } from 'react';

const LabelPanel = ({ rowIndex }) => {
  const [label, setLabel] = useState(null);

  const handleLabel = (val) => {
    setLabel(val);
    console.log(`Row ${rowIndex} labeled as: ${val}`);
    // TODO: Save this status to memory or backend
  };

  return (
    <div>
      <strong>Label:</strong>
      <button onClick={() => handleLabel(0)} style={{ marginLeft: 10 }}>
        Normal
      </button>
      <button onClick={() => handleLabel(1)} style={{ marginLeft: 10 }}>
        Abnormal
      </button>
      <span style={{ marginLeft: 20, fontWeight: 'bold' }}>
        Selected: {label === null ? 'None' : label === 0 ? 'Normal' : 'Abnormal'}
      </span>
    </div>
  );
};

export default LabelPanel;
