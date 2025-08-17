import React, { useState } from 'react';
import API from '../api/api';
import { useNavigate, Link } from 'react-router-dom';
import Navbar from '../components/Navbar';

const SignupPage = () => {
  const [form, setForm] = useState({ username: '', password: '' });
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleChange = e => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setMessage('');
    try {
      await API.post('/register/', form);
      setMessage('Registration successful! Redirecting to login...');
      setTimeout(() => navigate('/login'), 2000);
    } catch (err) {
      setError('Registration failed: ' + (err.response?.data?.username || 'Please try again.'));
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
            <h2 style={styles.title}>Sign Up</h2>
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
              autoComplete="new-password"
            />
            {error && <p style={styles.error}>{error}</p>}
            {message && <p style={styles.message}>{message}</p>}
            <button type="submit" style={styles.button} disabled={loading}>
              {loading ? 'Registering...' : 'Sign Up'}
            </button>
            <p style={styles.text}>
              Already have an account? <Link to="/login">Login</Link>
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
    minHeight: 'calc(100vh - 60px)', // Assuming navbar height approx 60px
    padding: 20,
    backgroundColor: '#fdfdfd',
  },
  formContainer: {
    width: '100%',
    maxWidth: 400,
    border: '1px solid #ddd',
    borderRadius: 8,
    padding: 24,
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
    backgroundColor: 'white',
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
    backgroundColor: '#28a745',
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
  message: {
    color: 'green',
    marginBottom: 16,
    fontSize: 14,
  },
  text: {
    marginTop: 16,
    fontSize: 14,
    textAlign: 'center',
  },
};

export default SignupPage;
