import React, { useState, useContext } from 'react';
import API from '../api/api';
import { useNavigate, Link } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { ThemeContext } from '../components/context/ThemeContext';

const SignupPage = () => {
  const { theme } = useContext(ThemeContext);
  const [form, setForm] = useState({ username: '', password: '', role: 'doctor', patient_id: '' });
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleChange = e => setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setMessage('');
    try {
      const payload = {
        username: form.username,
        password: form.password,
        role: form.role,
        ...(form.role === 'patient' ? { patient_id: form.patient_id } : {}),
      };
      await API.post('/register/', payload);
      setMessage('Registration successful! Redirecting to login...');
      setTimeout(() => navigate('/login'), 2000);
    } catch (err) {
      const data = err.response?.data;
      const firstFieldError = data && typeof data === 'object'
        ? Object.values(data).find((value) => Array.isArray(value) && value.length > 0)?.[0]
        : null;
      setError(
        data?.error ||
        data?.detail ||
        firstFieldError ||
        'Registration failed: Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Navbar />
      <div
        className="min-h-screen flex items-center justify-center bg-[var(--bg)] text-[var(--text)] px-6"
      >
        <div
          className="max-w-sm w-full bg-[var(--card-bg)] border border-[var(--border)] rounded-lg p-6 shadow-md"
          style={{ color: theme.text }}
        >
          <h2 className="text-2xl font-semibold text-center mb-6">Sign Up</h2>
          <form className="flex flex-col space-y-4" onSubmit={handleSubmit}>
            <input
              name="username"
              placeholder="Username"
              value={form.username}
              onChange={handleChange}
              required
              autoComplete="username"
              className="px-3 py-2 border border-[var(--border)] rounded focus:outline-none focus:ring-2 focus:ring-[var(--accent)] bg-[var(--card-bg)] text-[var(--text)] text-sm"
            />
            <input
              type="password"
              name="password"
              placeholder="Password"
              value={form.password}
              onChange={handleChange}
              required
              autoComplete="new-password"
              className="px-3 py-2 border border-[var(--border)] rounded focus:outline-none focus:ring-2 focus:ring-[var(--accent)] bg-[var(--card-bg)] text-[var(--text)] text-sm"
            />
            <select
              name="role"
              value={form.role}
              onChange={handleChange}
              className="px-3 py-2 border border-[var(--border)] rounded focus:outline-none focus:ring-2 focus:ring-[var(--accent)] bg-[var(--card-bg)] text-[var(--text)] text-sm"
            >
              <option value="doctor">Doctor</option>
              <option value="patient">Patient</option>
            </select>
            {form.role === 'patient' && (
              <input
                name="patient_id"
                placeholder="Patient ID"
                value={form.patient_id}
                onChange={handleChange}
                required
                className="px-3 py-2 border border-[var(--border)] rounded focus:outline-none focus:ring-2 focus:ring-[var(--accent)] bg-[var(--card-bg)] text-[var(--text)] text-sm"
              />
            )}
            {error && <p className="text-[var(--danger)] text-center text-sm">{error}</p>}
            {message && <p className="text-[var(--accent)] text-center text-sm">{message}</p>}
            <button
              type="submit"
              disabled={loading}
              className="bg-[var(--accent)] text-[var(--btn-text)] py-2 rounded font-semibold shadow hover:bg-[var(--accent-dark)] disabled:opacity-50 disabled:cursor-not-allowed transition text-sm"
            >
              {loading ? 'Registering...' : 'Sign Up'}
            </button>
            <p className="text-center mt-4 text-sm text-[var(--text)]">
              Already have an account?{' '}
              <Link to="/login" className="text-[var(--accent)] hover:underline font-medium">
                Login
              </Link>
            </p>
          </form>
        </div>
      </div>
    </>
  );
};

export default SignupPage;
