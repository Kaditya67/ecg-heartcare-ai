import React, { useEffect, useState, useContext } from 'react';
import { Pie } from 'react-chartjs-2';
import API from '../api/api';
import { ThemeContext } from '../components/context/ThemeContext';
import Footer from '../components/Footer'; // import your Footer component

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

  const pieData = {
    labels: Object.keys(metrics.labels_data),
    datasets: [{
      data: Object.values(metrics.labels_data),
      backgroundColor: [
        '#36a2eb', '#ff6384', '#ffce56', '#009688', '#8bc34a',
        '#f44336', '#ea80fc', '#03a9f4', '#ff9800', '#9c27b0'
      ],
    }]
  };

  const cardBg = theme === 'dark' ? '#23272f' : '#f6f6f6';
  const cardText = theme === 'dark' ? '#f0f0f0' : '#171923';

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
        backgroundColor: theme === 'dark' ? '#171923' : '#fff',
        color: theme === 'dark' ? '#f0f0f0' : '#171923',
        width: '100%',
        transition: 'background 0.3s, color 0.3s',
      }}
    >
      <main style={{ flexGrow: 1, width: '100%' }}>
        <div
          style={{
            maxWidth: 960,
            margin: 'auto',
            padding: 24,
          }}
        >
          <h2
            style={{
              textAlign: 'center',
              fontWeight: 700,
              fontSize: '2rem',
              marginBottom: 32,
            }}
          >
            ECG Dashboard
          </h2>
          <div
            style={{
              display: 'flex',
              gap: 24,
              marginBottom: 32,
              justifyContent: 'center',
              flexWrap: 'wrap',
            }}
          >
            <Card label="Total Files" value={metrics.total_files} bg={cardBg} text={cardText} />
            <Card label="Total Records" value={metrics.total_records} bg={cardBg} text={cardText} />
            <Card label="Labelled Records" value={metrics.labelled_records} bg={cardBg} text={cardText} />
            <Card label="Normal Records" value={metrics.normal_records} bg={cardBg} text={cardText} />
          </div>
          <div
            style={{
              maxWidth: 480,
              margin: 'auto',
              padding: 16,
              background: cardBg,
              borderRadius: 12,
              boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
            }}
          >
            <h4
              style={{
                textAlign: 'center',
                fontWeight: 600,
                marginBottom: 18,
                color: cardText,
              }}
            >
              Label Distribution
            </h4>
            <Pie data={pieData} />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

function Card({ label, value, bg, text }) {
  return (
    <div
      style={{
        padding: 24,
        background: bg,
        borderRadius: 10,
        boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
        minWidth: 140,
        textAlign: 'center',
        fontSize: 18,
        flex: 1,
        color: text,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{label}</div>
      <div style={{ fontSize: 32, color: '#36a2eb', fontWeight: 700 }}>{value}</div>
    </div>
  );
}

export default Dashboard;
