import React from 'react'; 
import DashboardNavbar from '../components/DashboardNavbar';
import Dashboard from './Dashboard';
import Footer from '../components/Footer';

const Home = () => {
  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
      <DashboardNavbar />
      <Dashboard />
    </div>
  );
};

export default Home;
