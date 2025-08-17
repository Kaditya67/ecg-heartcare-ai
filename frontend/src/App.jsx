import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Labeler from './pages/Labeler';
import LandingPage from './pages/LandingPage';
import UploadPage from './pages/UploadPage';
import PaginatedDataPage from './pages/PaginatedDataPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/dashboard" element={<Home />} />
      <Route path="/upload" element={<UploadPage />} />
      <Route path="/EcgLabel" element={<PaginatedDataPage />} />
      <Route path="/label" element={<Labeler />} />
    </Routes>
  );
}

export default App;
