// App.js or wherever your routes are defined
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import LandingPage from './pages/LandingPage';
import UploadPage from './pages/UploadPage';
import PaginatedDataPage from './pages/PaginatedDataPage';
import CustomExportPage from './pages/CustomExportPage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import ProtectedRoute from './components/ProtectedRoute';
import ModelPage from './pages/ModelPage';
import TrainingPage from './pages/TrainingPage';
import AnalysisPage from './pages/AnalysisPage';
import SettingsPage from './pages/SettingsPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />

      {/* Protected Routes */}
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute allowedRoles={['admin', 'doctor', 'patient']}>
            <Home />
          </ProtectedRoute>
        }
      />
      <Route
        path="/upload"
        element={
          <ProtectedRoute allowedRoles={['admin']}>
            <UploadPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/EcgLabel"
        element={
          <ProtectedRoute allowedRoles={['admin', 'doctor', 'patient']}>
            <PaginatedDataPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/custom"
        element={
          <ProtectedRoute allowedRoles={['admin', 'doctor', 'patient']}>
            <CustomExportPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/models"
        element={
          <ProtectedRoute allowedRoles={['admin', 'doctor', 'patient']}>
            <ModelPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/training"
        element={
          <ProtectedRoute allowedRoles={['admin']}>
            <TrainingPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/analysis"
        element={
          <ProtectedRoute allowedRoles={['admin', 'doctor']}>
            <AnalysisPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <ProtectedRoute allowedRoles={['admin', 'doctor', 'patient']}>
            <SettingsPage />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default App;
