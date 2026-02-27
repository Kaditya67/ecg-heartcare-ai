import React, { useEffect, useState, useContext, useMemo } from 'react';
import { Pie } from 'react-chartjs-2';
import API from '../api/api';
import { ThemeContext } from '../components/context/ThemeContext';
import Footer from '../components/Footer';
import ChatbotWidget from './ChatbotWidget';
import DashboardNavbar from '../components/DashboardNavbar';
import { FaFileMedical, FaWaveSquare, FaCheckDouble, FaRobot, FaDatabase, FaMicrochip } from 'react-icons/fa';

const Dashboard = () => {
  const { theme } = useContext(ThemeContext);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await API.get('/dashboard/summary/');
        if (res.status === 200) {
          setMetrics(res.data);
        } else {
          console.error('Failed to fetch metrics:', res.statusText);
        }
      } catch (err) {
        console.error('Error fetching metrics:', err);
      }
    };
    fetchMetrics();
  }, []);

  // Sync colors between Human and AI charts
  const labelColorMap = useMemo(() => {
    const defaultColors = [
      '#36a2eb', '#ff6384', '#ffce56', '#009688', '#8bc34a',
      '#f44336', '#ea80fc', '#03a9f4', '#ff9800', '#9c27b0'
    ];
    const map = {};
    if (metrics) {
      // Prioritize labels from human data for color assignment
      const allLabels = Array.from(new Set([
        ...Object.keys(metrics.labels_data),
        ...Object.keys(metrics.ai_labels_data)
      ]));
      allLabels.forEach((label, idx) => {
        map[label] = defaultColors[idx % defaultColors.length];
      });
    }
    return map;
  }, [metrics]);

  if (!metrics)
    return (
      <div
        style={{
          background: theme === 'dark' ? '#171923' : '#fff',
          color: theme === 'dark' ? '#f0f0f0' : '#171923',
          minHeight: '100vh',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        Loading...
      </div>
    );

  const humanPieData = {
    labels: Object.keys(metrics.labels_data),
    datasets: [{
      data: Object.values(metrics.labels_data),
      backgroundColor: Object.keys(metrics.labels_data).map(label => labelColorMap[label]),
    }]
  };

  const aiPieData = {
    labels: Object.keys(metrics.ai_labels_data),
    datasets: [{
      data: Object.values(metrics.ai_labels_data),
      backgroundColor: Object.keys(metrics.ai_labels_data).map(label => labelColorMap[label]),
    }]
  };

  return (
    <div className="flex flex-col bg-[var(--bg)] text-[var(--text)] transition-colors duration-300">
      
      <main className="flex-grow p-4 lg:p-10">
        <div className="max-w-[1200px] mx-auto space-y-10">
          
          <header className="flex flex-col items-center text-center space-y-1 border-b border-[var(--border)] pb-6">
            <h1 className="text-3xl font-bold tracking-tight">System Dashboard</h1>
            <p className="text-[10px] font-bold uppercase tracking-widest text-gray-500">Real-time Clinical Data Analytics</p>
          </header>

          <div className="space-y-10">
            {/* Metric Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              <MetricCard icon={<FaDatabase />} label="Total Files" value={metrics.total_files} />
              <MetricCard icon={<FaWaveSquare />} label="Total Records" value={metrics.total_records} />
              {metrics.labelled_records > 0 && <MetricCard icon={<FaFileMedical />} label="Labelled" value={metrics.labelled_records} color="text-green-500" />}
              {metrics.verified_records > 0 && <MetricCard icon={<FaCheckDouble />} label="Verified" value={metrics.verified_records} color="text-emerald-600" />}
              {metrics.ai_labelled_records > 0 && <MetricCard icon={<FaRobot />} label="AI Labeled" value={metrics.ai_labelled_records} color="text-purple-500" />}
              {metrics.total_records > 0 && (
                <MetricCard 
                  icon={<FaMicrochip />} 
                  label="Label Rate" 
                  value={`${Math.round((metrics.labelled_records / metrics.total_records) * 100) || 0}%`} 
                />
              )}
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {Object.keys(metrics.labels_data).length > 0 && (
                <div className="card space-y-6">
                  <div className="flex justify-between items-center border-b border-[var(--border)] pb-4">
                    <h3 className="text-xs font-semibold uppercase tracking-widest text-gray-500">Human Annotation Distribution</h3>
                    <span className="text-[10px] font-bold bg-[var(--highlight)] px-2 py-0.5 rounded-full border border-[var(--border)] text-gray-400">N={metrics.labelled_records}</span>
                  </div>
                  <div className="aspect-square max-w-[350px] mx-auto">
                    <Pie data={humanPieData} options={{ maintainAspectRatio: false, plugins: { legend: { labels: { font: { size: 10, weight: '600' } } } } }} />
                  </div>
                </div>
              )}

              {Object.keys(metrics.ai_labels_data).length > 0 && (
                <div className="card space-y-6">
                  <div className="flex justify-between items-center border-b border-[var(--border)] pb-4">
                    <h3 className="text-xs font-semibold uppercase tracking-widest text-gray-500">AI Prediction Distribution</h3>
                    <span className="text-[10px] font-bold bg-[var(--highlight)] px-2 py-0.5 rounded-full border border-[var(--border)] text-gray-400">N={metrics.ai_labelled_records}</span>
                  </div>
                  <div className="aspect-square max-w-[350px] mx-auto">
                    <Pie 
                      data={aiPieData} 
                      options={{ maintainAspectRatio: false, plugins: { legend: { labels: { font: { size: 10, weight: '600' } } } } }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <Footer />
      <ChatbotWidget />
    </div>
  );
};

const MetricCard = ({ icon, label, value, color="" }) => (
  <div className="card text-center p-6 flex flex-col items-center justify-center space-y-1.5 hover:border-[var(--accent)] transition-all group">
    <div className={`text-xl opacity-30 group-hover:opacity-100 transition-opacity ${color || 'text-gray-400'}`}>{icon}</div>
    <div className="text-[9px] font-medium uppercase tracking-widest text-gray-500">{label}</div>
    <div className="text-2xl font-semibold">{value}</div>
  </div>
);

export default Dashboard;
