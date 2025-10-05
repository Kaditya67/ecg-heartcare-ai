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
          <ProtectedRoute>
            <Home />
          </ProtectedRoute>
        }
      />
      <Route
        path="/upload"
        element={
          <ProtectedRoute>
            <UploadPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/EcgLabel"
        element={
          <ProtectedRoute>
            <PaginatedDataPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/custom"
        element={
          <ProtectedRoute>
            <CustomExportPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/models"
        element={
          <ProtectedRoute>
            <ModelPage />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default App;
