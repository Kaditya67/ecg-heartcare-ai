import axios from 'axios';

const API = axios.create({
  baseURL: 'http://localhost:8000/api',
});

// Add a request interceptor to include the token in headers automatically
API.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('accessToken'); // or wherever you store your token
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export default API;
