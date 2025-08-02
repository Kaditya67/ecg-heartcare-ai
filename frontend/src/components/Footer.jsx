import React from 'react';

const Footer = () => {
  return (
    <footer className="py-4 text-center text-sm border-t border-[var(--border)]">
      <p>&copy; {new Date().getFullYear()} ECG Labeling System. All rights reserved.</p>
    </footer>
  );
};

export default Footer;