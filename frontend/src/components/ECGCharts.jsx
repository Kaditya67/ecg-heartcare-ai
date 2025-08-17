import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Line } from 'react-chartjs-2';
import ReactECharts from 'echarts-for-react';
import 'chart.js/auto';

const MAX_PLOT_SAMPLES = 5000;

function downsampleECG(arr) {
  if (!arr) return [];
  if (arr.length <= MAX_PLOT_SAMPLES) return arr;
  const skip = Math.ceil(arr.length / MAX_PLOT_SAMPLES);
  return arr.filter((_, idx) => idx % skip === 0);
}

function getVar(name, fallback) {
  if (typeof window !== "undefined") {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  }
  return fallback;
}

const ECGCharts = ({
  ecgArray,
  chartType,
  style = {},
  loading = false,
  error = null,
  maxSamples = MAX_PLOT_SAMPLES,
  theme // <--- receives theme from props
}) => {
  if (loading) return <div className="text-blue-500 py-10 text-center">Loading chart…</div>;
  if (error) return <div className="text-red-600 py-10 text-center">Failed to load chart: {error}</div>;
  if (!ecgArray || !ecgArray.length) return <div className="text-gray-500 text-center">No ECG data.</div>;

  // Recompute colors whenever the theme or chartType changes
  const { accentColor, textColor, gridColor, cardBg, highlight } = useMemo(() => ({
    accentColor: getVar('--accent', '#2563eb'),
    textColor: getVar('--text', '#222'),
    gridColor: getVar('--border', '#e5e7eb'),
    cardBg: getVar('--card-bg', '#fff'),
    highlight: getVar('--highlight', '#f1f5f9')
  }), [theme, chartType]);

  const ecg = maxSamples ? downsampleECG(ecgArray) : ecgArray;
  const sampleIndices = Array.from({ length: ecg.length }, (_, i) => i);

  if (chartType === "plotly") {
    return (
      <Plot
        data={[
          {
            x: sampleIndices,
            y: ecg,
            type: 'scatter',
            mode: 'lines',
            line: { color: accentColor, width: 2 },
          },
        ]}
        layout={{
          autosize: true,
          height: 380,
          paper_bgcolor: cardBg,
          plot_bgcolor: cardBg,
          font: { color: textColor, size: 14, family: 'inherit' },
          margin: { l: 50, r: 30, t: 30, b: 50 },
          xaxis: {
            title: 'Length',
            zeroline: false,
            showgrid: true,
            gridcolor: gridColor,
            color: textColor,
            linecolor: gridColor,
            tickcolor: gridColor,
          },
          yaxis: {
            title: 'ECG Value',
            zeroline: false,
            showgrid: true,
            gridcolor: gridColor,
            color: textColor,
            linecolor: gridColor,
            tickcolor: gridColor,
          },
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: "100%", ...style }}
      />
    );
  }

  if (chartType === "chartjs") {
    return (
      <div style={{ width: "100%", minHeight: 320, ...style }}>
        <Line
          data={{
            labels: sampleIndices,
            datasets: [{
              label: 'ECG',
              data: ecg,
              fill: false,
              borderColor: accentColor,
              backgroundColor: highlight,
              borderWidth: 2,
              tension: 0.2,
              pointRadius: 0,
            }]
          }}
          height={380}
          options={{
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
            },
            layout: {
              padding: 10,
            },
            scales: {
              x: {
                title: { display: true, text: 'Length', color: textColor },
                grid: { color: gridColor },
                ticks: { color: textColor },
              },
              y: {
                title: { display: true, text: 'ECG Value', color: textColor },
                grid: { color: gridColor },
                ticks: { color: textColor },
              },
            },
            animation: false,
            backgroundColor: cardBg,
          }}
        />
      </div>
    );
  }

  if (chartType === "echarts") {
    return (
      <ReactECharts
        style={{ width: "100%", minHeight: 380, ...style }}
        option={{
          animation: false,
          tooltip: { trigger: 'axis' },
          grid: { left: 60, right: 30, top: 30, bottom: 50 },
          backgroundColor: cardBg,
          xAxis: {
            type: 'category',
            data: sampleIndices,
            name: 'Length',
            nameLocation: 'center',
            nameGap: 32,
            axisLine: { lineStyle: { color: textColor } },
            axisLabel: { fontSize: 13, color: textColor },
            splitLine: { lineStyle: { color: gridColor } },
          },
          yAxis: {
            type: 'value',
            name: 'ECG Value',
            nameLocation: 'center',
            nameGap: 40,
            axisLine: { lineStyle: { color: textColor } },
            axisLabel: { fontSize: 13, color: textColor },
            splitLine: { lineStyle: { color: gridColor } },
          },
          series: [{
            data: ecg,
            type: 'line',
            smooth: true,
            showSymbol: false,
            lineStyle: { color: accentColor, width: 2 },
          }]
        }}
      />
    );
  }

  return <div className="text-gray-500 text-center">Select a chart type.</div>;
};

export default ECGCharts;
