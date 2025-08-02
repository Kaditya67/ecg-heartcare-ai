import React from 'react';
import Navbar from '../components/Navbar';
import HeroSection from '../components/HeroSection';
import FeaturesSection from '../components/FeaturesSection';
import DeveloperSection from '../components/DeveloperSection';
import ContactSection from '../components/ContactSection';
import Footer from '../components/Footer';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text)] font-sans">
      <Navbar />
      <HeroSection />
      <FeaturesSection />
      <DeveloperSection />
      <ContactSection />
      <Footer />
    </div>
  );
};

export default LandingPage;
