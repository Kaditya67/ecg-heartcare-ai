import React from 'react';
import Navbar from '../components/Navbar';
import HeroSection from '../components/HeroSection';
import FeaturesSection from '../components/FeaturesSection';
import DeveloperSection from '../components/DeveloperSection';
import ContactSection from '../components/ContactSection';
import Footer from '../components/Footer';

const LandingPage = () => (
  <div className="min-h-screen bg-[var(--bg)] text-[var(--text)] font-sans">
    <Navbar />
    <HeroSection />                 {/* Only Lottie used here */}
    <FeaturesSection />             {/* Use SVGs/PNGs here */}
    <DeveloperSection />            {/* Use SVG or avatar PNG here */}
    <ContactSection />
    <Footer />
  </div>
);

export default LandingPage;
