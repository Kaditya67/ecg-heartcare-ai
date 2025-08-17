import React from 'react';

const ContactSection = () => (
  <section className="bg-[var(--card-bg)] py-20 px-6 border-t border-[var(--border)]">
    <div className="max-w-3xl mx-auto text-center">
      <h3 className="text-3xl sm:text-4xl font-extrabold mb-6 text-[var(--text)]">
        Get in Touch
      </h3>

      <p className="text-base sm:text-lg text-[var(--text)] opacity-90 mb-6">
        Have questions or ideas? Let's connect and explore how we can collaborate on meaningful ECG and healthtech solutions.
      </p>

      <p className="text-[var(--text)] font-medium mb-6">
        Email: <a href="mailto:ojhaaditya913@gmail.com" className="text-[var(--accent)] hover:underline">
          ojhaaditya913@gmail.com
        </a>
      </p>

      <a href="mailto:ojhaaditya913@gmail.com" aria-label="Send email to ojhaaditya913@gmail.com">
        <button className="bg-[var(--secondary)] text-[var(--text)] font-medium px-6 py-3 rounded-lg shadow hover:opacity-90 transition">
          Contact Me
        </button>
      </a>
    </div>
  </section>
);

export default ContactSection;
