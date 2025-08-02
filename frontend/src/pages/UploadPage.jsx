import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import DashboardNavbar from '../components/DashboardNavbar';

const UploadPage = () => {
  const [files, setFiles] = useState([]);

  const handleFileUpload = (newFile, preview) => {
    const newEntry = {
      id: Date.now(),
      name: newFile.name,
      size: (newFile.size / 1024).toFixed(2) + ' KB',
      uploadedAt: new Date().toLocaleString(),
      preview, // first 5 rows
    };
    setFiles([newEntry, ...files]);
  };

  const handleDelete = (id) => {
    setFiles(prev => prev.filter(file => file.id !== id));
  };

  return (
    <>
    <DashboardNavbar />
    <section className="p-6 max-w-4xl mx-auto space-y-6">
      <h2 className="text-2xl font-semibold">Upload ECG Files</h2>
      <FileUpload onUpload={handleFileUpload} />

      <div className="space-y-6">
        {files.length === 0 ? (
          <p className="text-sm text-gray-500">No files uploaded yet.</p>
        ) : (
          files.map(file => (
            <div key={file.id} className="card space-y-3">
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-medium">{file.name}</p>
                  <p className="text-sm text-gray-500">{file.size} • {file.uploadedAt}</p>
                </div>
                <button
                  className="button-abnormal"
                  onClick={() => handleDelete(file.id)}
                >
                  Delete
                </button>
              </div>

              {/* Preview Table */}
              {file.preview && file.preview.length > 0 && (
                <div className="overflow-auto">
                  <table className="min-w-full text-sm border border-[var(--border)]">
                    <thead>
                      <tr>
                        {Object.keys(file.preview[0]).map((key, idx) => (
                          <th key={idx} className="border px-3 py-1 bg-[var(--highlight)]">{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {file.preview.map((row, idx) => (
                        <tr key={idx}>
                          {Object.values(row).map((val, i) => (
                            <td key={i} className="border px-3 py-1">{val || '-'}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </section>
    </>
  );
};

export default UploadPage;
