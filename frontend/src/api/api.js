import axios from 'axios';

const API = axios.create({
  baseURL: '/api',
});

let refreshPromise = null;

const serializeRequestData = (data) => {
  if (data instanceof FormData) {
    return Array.from(data.entries()).map(([key, value]) => {
      if (value instanceof File) {
        return [key, { name: value.name, size: value.size, type: value.type }];
      }
      return [key, value];
    });
  }
  return data;
};

const clearAuthSession = () => {
  localStorage.removeItem('accessToken');
  localStorage.removeItem('refreshToken');
  localStorage.removeItem('authUser');
  localStorage.removeItem('authRole');
};

const refreshAccessToken = async () => {
  const refreshToken = localStorage.getItem('refreshToken');
  if (!refreshToken) {
    throw new Error('No refresh token available');
  }

  if (!refreshPromise) {
    refreshPromise = axios.post('/api/token/refresh/', { refresh: refreshToken })
      .then((response) => {
        const nextAccessToken = response.data?.access;
        if (!nextAccessToken) {
          throw new Error('Refresh response did not include access token');
        }
        localStorage.setItem('accessToken', nextAccessToken);
        return nextAccessToken;
      })
      .finally(() => {
        refreshPromise = null;
      });
  }

  return refreshPromise;
};

// Add a request interceptor to include the token in headers automatically
API.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('accessToken'); // or wherever you store your token
    const url = config.url || '';
    const isPublicAuthRoute = url.includes('/login/') || url.includes('/register/');
    if (token && !isPublicAuthRoute) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    if (typeof window !== 'undefined') {
      const method = (config.method || 'get').toUpperCase();
      console.info(
        '[API request]',
        method,
        config.baseURL ? `${config.baseURL}${config.url}` : config.url,
        serializeRequestData(config.data || '')
      );
    }
    return config;
  },
  (error) => Promise.reject(error)
);

API.interceptors.response.use(
  (response) => {
    if (typeof window !== 'undefined') {
      const method = (response.config?.method || 'get').toUpperCase();
      console.info('[API response]', method, response.config?.url, response.status, response.data || '');
    }
    return response;
  },
  async (error) => {
    if (typeof window !== 'undefined') {
      const method = (error.config?.method || 'get').toUpperCase();
      console.error('[API error]', method, error.config?.url, error.response?.status, error.response?.data || error.message);
    }

    const originalRequest = error.config;
    const status = error.response?.status;
    const url = originalRequest?.url || '';
    const isRefreshCall = url.includes('/token/refresh/');
    const isLoginCall = url.includes('/login/');

    if (status === 401 && originalRequest && !originalRequest._retry && !isRefreshCall && !isLoginCall) {
      originalRequest._retry = true;
      try {
        const nextAccessToken = await refreshAccessToken();
        originalRequest.headers = originalRequest.headers || {};
        originalRequest.headers.Authorization = `Bearer ${nextAccessToken}`;
        return API(originalRequest);
      } catch (refreshError) {
        clearAuthSession();
        if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
          window.location.assign('/login');
        }
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

export default API;
