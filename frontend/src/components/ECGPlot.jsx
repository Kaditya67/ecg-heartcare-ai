import React from 'react';
import Plot from 'react-plotly.js';

const ECGPlot = ({ row, index }) => {
  const ecgPoints = [];

  // Assuming you stored comma-separated ECG data in one column named "ECG"
  if (row['ECG']) {
    const raw = row['ECG'].split(',').map(Number);
    for (let i = 0; i < raw.length; i++) {
      ecgPoints.push({ x: i, y: raw[i] });
    }
  }

  return (
    <div>
      <h4>Row #{index + 1}</h4>
      <Plot
        data={[
          {
            x: ecgPoints.map(p => p.x),
            y: ecgPoints.map(p => p.y),
            type: 'scatter',
            mode: 'lines',
            marker: { color: 'blue' },
          },
        ]}
        layout={{
          width: 600,
          height: 200,
          margin: { l: 30, r: 30, t: 30, b: 30 },
        }}
      />
    </div>
  );
};

export default ECGPlot;
