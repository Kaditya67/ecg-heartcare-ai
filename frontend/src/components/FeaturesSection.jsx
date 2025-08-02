import React, { useState } from 'react';
import Lottie from 'lottie-react';

import uploadAnim from '../assets/animations/upload.json';
import labelAnim from '../assets/animations/label.json';
import progressAnim from '../assets/animations/progress.json';
import LazyLottie from './LazyLottie';

const features = [
  {
    title: 'Upload & View',
    desc: 'Seamlessly upload Excel sheets and instantly visualize ECG plots with intuitive, zoomable graphs. No setup required.',
    animation: uploadAnim,
  },
  {
    title: 'Smart Labeling',
    desc: 'Label ECG data using predefined sets or custom tags. Tailor it to your research or diagnostic needs with ease.',
    animation: labelAnim,
  },
  {
    title: 'Session Progress',
    desc: 'Track all actions—what’s labeled, skipped, or pending. Stay organized and resume effortlessly.',
    animation: progressAnim,
  },
];

const FeaturesSection = () => {
  const [selectedIndex, setSelectedIndex] = useState(null);

  return (
    <section className="py-24 px-4 sm:px-8 bg-[var(--highlight)] text-[var(--text)] relative z-10">
      <div className="max-w-7xl mx-auto">
        <h3 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-center mb-20">
          What You Can Do
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-12">
          {features.map((feature, idx) => (
            <div
              key={idx}
              onClick={() => setSelectedIndex(idx)}
              className={`relative cursor-pointer bg-[var(--card-bg)] border transition-all duration-500 border-[var(--border)] rounded-2xl shadow-lg overflow-hidden p-6 sm:p-8 text-center flex flex-col items-center backdrop-blur-xl group
                ${selectedIndex === idx ? 'ring-2 ring-[var(--accent)] scale-[1.03]' : ''}
                hover:scale-[1.03] hover:shadow-2xl hover:border-[var(--accent)]`}
            >
              {/* Animation */}
              <div className="w-full h-44 sm:h-48 max-w-[240px] z-10 mb-6">
                <LazyLottie animationData={feature.animation} loop autoplay aria-label={`${feature.title} animation`} />
              </div>

              {/* Title */}
              <h4 className="text-xl font-semibold mb-3 z-10">{feature.title}</h4>

              {/* Description */}
              <p className="text-sm sm:text-base opacity-90 leading-relaxed z-10">
                {feature.desc}
              </p>

              {/* Glow border on hover */}
              <div className="absolute inset-0 z-0 pointer-events-none rounded-2xl transition-all duration-300 group-hover:ring-4 group-hover:ring-[var(--accent)] group-hover:ring-opacity-10"></div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
