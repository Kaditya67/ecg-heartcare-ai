import React from 'react';
import Lottie from 'lottie-react';
import devAnim from '../assets/animations/dev-avatar.json';

const DeveloperSection = () => {
  return (
    <section className="py-24 px-6 bg-[var(--card-bg)] border-t border-[var(--border)] text-center text-[var(--text)]">
      <div className="max-w-3xl mx-auto">
        {/* Lottie Avatar */}
        <div className="w-28 h-28 sm:w-32 sm:h-32 mx-auto mb-6 rounded-full overflow-hidden shadow-lg">
          <Lottie animationData={devAnim} loop autoplay className="scale-[1.1]" />
        </div>

        <h3 className="text-3xl sm:text-4xl font-extrabold mb-4">
          Meet the Developer
        </h3>

        <p className="text-base sm:text-lg opacity-90 leading-relaxed px-2">
          Built with <span className="text-[var(--danger)] font-semibold">passion</span> by
          <span className="text-[var(--accent)] font-semibold"> Aditya Ojha</span> — a developer dedicated to building clean, impactful healthtech systems.
          <br className="hidden sm:block" />
          Let’s collaborate and shape the future of ECG intelligence together.
        </p>
      </div>
    </section>
  );
};

export default DeveloperSection;
