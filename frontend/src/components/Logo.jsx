import React, { useEffect, useState } from 'react';

const Logo = ({ className = 'h-8' }) => {
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    const observer = new MutationObserver(() => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      setTheme(currentTheme || 'light');
    });

    observer.observe(document.documentElement, { attributes: true });
    setTheme(document.documentElement.getAttribute('data-theme') || 'light');

    return () => observer.disconnect();
  }, []);

  const logoSrc = '/logo.png'; // Your single logo
  const filterClass =
    theme === 'dark'
      ? 'invert brightness-[5] drop-shadow-md'
      : 'brightness-[0.9] drop-shadow';

  return (
    <img
      src={logoSrc}
      alt="ECG Labeling System"
      className={`transition-all duration-300 ${filterClass} ${className}`}
    />
  );
};

export default Logo;
