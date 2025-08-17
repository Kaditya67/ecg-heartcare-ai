// components/ProtectedRoute.js
import React from 'react';
import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ children }) => {
  const isAuthenticated = !!localStorage.getItem('accessToken'); // Check if logged in

  if (!isAuthenticated) {
    // Redirect non-authenticated users to login page
    return <Navigate to="/login" replace />;
  }

  return children;
};

export default ProtectedRoute;
