import React, { useEffect, useState, useContext } from 'react';
import { Link, useLocation } from 'react-router-dom';
import Logo from './Logo';
import { ThemeContext } from './context/ThemeContext';

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

const DashboardNavbar = () => {
  const { theme, toggleTheme } = useContext(ThemeContext);
  const location = useLocation();

  const navLinks = [
    { name: 'Dashboard', path: '/dashboard' },
    { name: 'Upload', path: '/upload' },
    { name: 'EcgLabel', path: '/EcgLabel' },
    { name: 'Stats', path: '/stats' },
  ];

  return (
    <nav className="w-full px-6 py-3 flex justify-between items-center bg-[var(--card-bg)] border-b border-[var(--border)]">
      {/* Logo and title */}
      <div className="flex items-center gap-2 text-xl font-bold">
        <Logo className="h-8" />
        ECG Dashboard
      </div>

      {/* Navigation Links */}
      <div className="hidden md:flex gap-6 text-sm font-medium">
        {navLinks.map(link => (
          <Link
            key={link.name}
            to={link.path}
            className={`hover:text-[var(--accent)] transition ${
              location.pathname === link.path ? 'text-[var(--accent)]' : 'text-[var(--text)]'
            }`}
          >
            {link.name}
          </Link>
        ))}
      </div>

      {/* Theme toggle and profile */}
      <div className="flex items-center gap-4">
        <button
          onClick={toggleTheme}
          className="p-2 border border-[var(--border)] rounded hover:bg-[var(--border)] transition"
          title="Toggle Theme"
        >
          {theme === 'dark' ? <MoonIcon /> : <SunIcon />}
        </button>

        <div className="flex items-center gap-2 cursor-pointer">
          <div className="h-8 w-8 rounded-full bg-[var(--accent)] text-white flex items-center justify-center font-semibold">
            A
          </div>
          <span className="text-sm font-medium hidden sm:inline">Aditya</span>
        </div>
      </div>
    </nav>
  );
};

export default DashboardNavbar;
