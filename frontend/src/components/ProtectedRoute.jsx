// components/ProtectedRoute.js
import React from 'react';
import { Navigate } from 'react-router-dom';
import { getStoredRole } from '../utils/auth';

const ProtectedRoute = ({ children, allowedRoles }) => {
  const isAuthenticated = !!localStorage.getItem('accessToken'); // Check if logged in
  const role = getStoredRole();

  if (!isAuthenticated) {
    // Redirect non-authenticated users to login page
    return <Navigate to="/login" replace />;
  }

  if (allowedRoles && allowedRoles.length > 0 && !allowedRoles.includes(role)) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
};

export default ProtectedRoute;
