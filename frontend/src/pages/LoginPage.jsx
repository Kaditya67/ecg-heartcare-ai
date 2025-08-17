import React, { useState } from 'react';
import API from '../api/api';
import { useNavigate, Link } from 'react-router-dom';
import Navbar from '../components/Navbar';

const LoginPage = () => {
  const [form, setForm] = useState({ username: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = e => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await API.post('/login/', form);
      localStorage.setItem('accessToken', res.data.access);
      localStorage.setItem('refreshToken', res.data.refresh);
      navigate('/dashboard'); // Redirect after login
    } catch (err) {
      setError('Login failed: Invalid username or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Navbar />
      <div style={styles.pageContainer}>
        <div style={styles.formContainer}>
          <form onSubmit={handleSubmit} style={styles.form}>
            <h2 style={styles.title}>Login</h2>
            <input
              name="username"
              placeholder="Username"
              value={form.username}
              onChange={handleChange}
              required
              style={styles.input}
              autoComplete="username"
            />
            <input
              type="password"
              name="password"
              placeholder="Password"
              value={form.password}
              onChange={handleChange}
              required
              style={styles.input}
              autoComplete="current-password"
            />
            {error && <p style={styles.error}>{error}</p>}
            <button type="submit" style={styles.button} disabled={loading}>
              {loading ? 'Logging in...' : 'Login'}
            </button>
            <p style={styles.text}>
              Don&apos;t have an account? <Link to="/signup">Sign Up</Link>
            </p>
          </form>
        </div>
      </div>
    </>
  );
};

const styles = {
  pageContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: 'calc(100vh - 60px)', // Adjust 60px if navbar height differs
    padding: 20,
    backgroundColor: '#fdfdfd',
  },
  formContainer: {
    width: '100%',
    maxWidth: 400,
    backgroundColor: 'white',
    padding: 24,
    borderRadius: 8,
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
    border: '1px solid #ddd',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
  },
  title: {
    marginBottom: 24,
    textAlign: 'center',
    color: '#333',
  },
  input: {
    marginBottom: 16,
    padding: 12,
    fontSize: 16,
    borderRadius: 4,
    border: '1px solid #ccc',
  },
  button: {
    padding: 12,
    fontSize: 16,
    backgroundColor: '#007BFF',
    color: '#fff',
    border: 'none',
    borderRadius: 4,
    cursor: 'pointer',
  },
  error: {
    color: 'crimson',
    marginBottom: 16,
    fontSize: 14,
  },
  text: {
    marginTop: 16,
    fontSize: 14,
    textAlign: 'center',
  },
};

export default LoginPage;
