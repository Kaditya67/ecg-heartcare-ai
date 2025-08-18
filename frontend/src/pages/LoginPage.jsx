import React, { useState, useContext } from 'react';
import API from '../api/api';
import { useNavigate, Link } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { ThemeContext } from '../components/context/ThemeContext';

const LoginPage = () => {
  const { theme } = useContext(ThemeContext);
  const [form, setForm] = useState({ username: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = e => setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      console.log('Logging in with:', form);
      const res = await API.post('/login/', form);
      localStorage.setItem('accessToken', res.data.access);
      localStorage.setItem('refreshToken', res.data.refresh);
      navigate('/dashboard');
    } catch {
      setError('Login failed: Invalid username or password');
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
          <h2 className="text-2xl font-semibold text-center mb-6">Login</h2>
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
              autoComplete="current-password"
              className="px-3 py-2 border border-[var(--border)] rounded focus:outline-none focus:ring-2 focus:ring-[var(--accent)] bg-[var(--card-bg)] text-[var(--text)] text-sm"
            />
            {error && <p className="text-[var(--danger)] text-center text-sm">{error}</p>}
            <button
              type="submit"
              disabled={loading}
              className="bg-[var(--accent)] text-[var(--btn-text)] py-2 rounded font-semibold shadow hover:bg-[var(--accent-dark)] disabled:opacity-50 disabled:cursor-not-allowed transition text-sm"
            >
              {loading ? 'Logging in...' : 'Login'}
            </button>
            <p className="text-center mt-4 text-sm text-[var(--text)]">
              Don't have an account?{' '}
              <Link to="/signup" className="text-[var(--accent)] hover:underline font-medium">
                Sign Up
              </Link>
            </p>
          </form>
        </div>
      </div>
    </>
  );
};

export default LoginPage;
