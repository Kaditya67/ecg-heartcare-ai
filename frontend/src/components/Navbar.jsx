import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import Logo from '../components/Logo';

const SunIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none"
    viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M12 3v2m0 14v2m9-9h-2M5 12H3m15.364 6.364l-1.414-1.414M6.05 6.05L4.636 4.636m0 14.728l1.414-1.414m12.728-12.728l-1.414 1.414M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
  </svg>
);

const MoonIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="currentColor"
    viewBox="0 0 20 20">
    <path d="M17.293 13.293a8 8 0 01-10.586-10.586 8 8 0 1010.586 10.586z" />
  </svg>
);

const Navbar = () => {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  return (
    <nav className="fixed top-0 left-0 w-full z-50 flex justify-between items-center px-6 py-3 backdrop-blur-md bg-[var(--card-bg)]/70 border-b border-[var(--border)] shadow-sm">
      {/* Logo */}
      <h1 className="text-xl font-semibold tracking-tight flex items-center gap-2">
        <Logo className="h-8" />
        ECG Labeling
      </h1>

      {/* Actions */}
      <div className="flex items-center gap-3 text-sm">
        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 border border-[var(--border)] rounded hover:bg-[var(--border)] transition"
          title="Toggle Theme"
        >
          {theme === 'dark' ? <MoonIcon /> : <SunIcon />}
        </button>

        {/* Auth Buttons */}
        <Link to="/login">
          <button className="px-4 py-1.5 rounded border border-[var(--border)] text-[var(--text)] hover:bg-[var(--border)] transition">
            Login
          </button>
        </Link>
        <Link to="/signup">
          <button className="px-4 py-1.5 rounded border border-[var(--accent)] text-[var(--accent)] hover:bg-[var(--accent)] hover:text-[var(--btn-text)] transition">
            Sign Up
          </button>
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
