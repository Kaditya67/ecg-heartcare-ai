import React, { useState, useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import API from '../api/api';
import DashboardNavbar from '../components/DashboardNavbar';
import Footer from '../components/Footer';
import { updateStoredUserProfile } from '../utils/auth';

const SettingsPage = () => {
  const [settings, setSettings] = useState({ show_missing_weights: true, default_plot_library: 'echarts' });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const fetchSettings = async () => {
      setLoading(true);
      try {
        const response = await API.get('/profile/settings/');
        setSettings(response.data || { show_missing_weights: true, default_plot_library: 'echarts' });
      } catch (err) {
        toast.error('Unable to load saved settings.');
      } finally {
        setLoading(false);
      }
    };
    fetchSettings();
  }, []);

  const handleSaveSettings = async () => {
    setSaving(true);
    try {
      const response = await API.patch('/profile/settings/', settings);
      setSettings(response.data);
      updateStoredUserProfile(response.data);
      toast.success('Settings saved successfully.');
    } catch (err) {
      toast.error(err.response?.data?.error || 'Failed to save settings.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[var(--bg)] text-[var(--text)] transition-colors duration-300">
      <DashboardNavbar />

      <main className="flex-grow container mx-auto px-4 py-10 max-w-4xl">
        <div className="space-y-8">
          <header className="mb-6">
            <h1 className="text-3xl font-bold tracking-tight">Profile Settings</h1>
            <p className="text-sm text-gray-500 mt-2">Persist your default preferences for model visibility and plotting behavior.</p>
          </header>

          <div className="card p-8 space-y-8">
            <section className="space-y-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-xl font-bold">Model Visibility</h2>
                  <p className="text-xs text-gray-500">Choose whether models missing weights should appear in the local registry.</p>
                </div>
                <label className="inline-flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={settings.show_missing_weights}
                    onChange={(e) => setSettings((prev) => ({ ...prev, show_missing_weights: e.target.checked }))}
                    className="h-4 w-4 rounded border-gray-300 text-[var(--accent)] focus:ring-[var(--accent)]"
                  />
                  <span className="font-semibold">Show missing-weight models</span>
                </label>
              </div>
            </section>

            <section className="space-y-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-xl font-bold">Default Plot Library</h2>
                  <p className="text-xs text-gray-500">Pick your preferred rendering style for model analytics.</p>
                </div>
                <select
                  value={settings.default_plot_library}
                  onChange={(e) => setSettings((prev) => ({ ...prev, default_plot_library: e.target.value }))}
                  className="bg-[var(--highlight)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm"
                >
                  <option value="echarts">ECharts heatmap</option>
                  <option value="table">HTML confusion table</option>
                </select>
              </div>
            </section>

            <div className="pt-6 border-t border-[var(--border)] flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div className="space-y-1 text-sm text-gray-500">
                <p className="font-semibold">Saved preferences</p>
                <p className="text-xs">These values are stored in your profile and applied automatically when loading the Models and Analysis pages.</p>
              </div>
              <button
                onClick={handleSaveSettings}
                disabled={saving || loading}
                className="px-5 py-3 bg-[var(--accent)] text-white rounded-lg font-bold uppercase tracking-widest disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save Settings'}
              </button>
            </div>
          </div>
        </div>
      </main>

      <Footer />
      <ToastContainer position="top-right" autoClose={3000} hideProgressBar={false} />
    </div>
  );
};

export default SettingsPage;
