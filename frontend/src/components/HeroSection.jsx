import React from 'react';
import { Link } from 'react-router-dom';
import Lottie from 'lottie-react';
import ecgAnimation from '../assets/animations/ecg-heartbeat.json';

const HeroSection = () => {
  return (
    <section className="relative h-screen flex items-center justify-center px-6 bg-[var(--bg)] text-[var(--text)]">
      <div className="max-w-7xl mx-auto flex flex-col-reverse md:flex-row items-center gap-12">
        {/* Text */}
        <div className="text-center md:text-left md:w-1/2 space-y-6">
          <h1 className="text-4xl md:text-5xl font-extrabold leading-tight">
            Welcome to <span className="text-[var(--accent)]">ECG Labeling System</span>
          </h1>
          <p className="text-lg opacity-90 leading-relaxed">
            Label ECG data effortlessly, organize your workflow, and get closer to accurate diagnosis and research insights.
          </p>
          <Link to="/dashboard">
            <button className="mt-4 bg-[var(--accent)] text-[var(--btn-text)] text-base md:text-lg px-6 py-3 rounded-lg shadow-md hover:opacity-90 transition">
              Explore
            </button>
          </Link>
        </div>
        {/* Animation */}
        <div className="md:w-3/4 w-full max-w-md">
          <Lottie key={Date.now()} animationData={JSON.parse(JSON.stringify(ecgAnimation))} loop autoplay />
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
