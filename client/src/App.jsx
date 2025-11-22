import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_URL = 'http://localhost:5000';

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      if (!file.type.match('image/(jpeg|jpg|png)')) {
        setError('Please select a valid image file (JPG, JPEG, or PNG)');
        return;
      }

      // Validate file size (16 MB max)
      if (file.size > 16 * 1024 * 1024) {
        setError('File size must be less than 16 MB');
        return;
      }

      setSelectedFile(file);
      setError(null);
      setResults(null);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post(`${API_URL}/api/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      });

      if (response.data.success) {
        setResults(response.data);
      } else {
        setError(response.data.error || 'Analysis failed');
      }
    } catch (err) {
      console.error('Error:', err);
      if (err.response) {
        setError(err.response.data.error || 'Server error');
      } else if (err.request) {
        setError('Cannot connect to server. Make sure Flask is running on port 5000');
      } else {
        setError('Error: ' + err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
  };

  const getSeverityColor = (score) => {
    if (score <= 10) return '#10b981'; // Green
    if (score <= 35) return '#f59e0b'; // Yellow
    if (score <= 70) return '#f97316'; // Orange
    return '#ef4444'; // Red
  };

  const getStageIcon = (diagnosis) => {
    const icons = {
      'No Impairment': '✅',
      'Very Mild Impairment': '⚠️',
      'Mild Impairment': '🔶',
      'Moderate Impairment': '🔴'
    };
    return icons[diagnosis] || '❓';
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <h1>🧠 Alzheimer's Detection System</h1>
        <p>AI-Powered MRI Scan Analysis</p>
      </header>

      <div className="container">
        {!results ? (
          /* Upload Section */
          <div className="upload-section">
            <div className="upload-card">
              <h2>Upload MRI Scan</h2>
              <p className="subtitle">Select a brain MRI image for analysis</p>

              <div className="upload-area">
                <input
                  type="file"
                  id="file-input"
                  accept="image/jpeg,image/jpg,image/png"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                />
                <label htmlFor="file-input" className="upload-label">
                  {preview ? (
                    <div className="preview-container">
                      <img src={preview} alt="Preview" className="preview-image" />
                      <div className="preview-overlay">
                        <span>Click to change image</span>
                      </div>
                    </div>
                  ) : (
                    <div className="upload-placeholder">
                      <div className="upload-icon">📁</div>
                      <p>Click to select MRI image</p>
                      <span className="upload-hint">Supports: JPG, JPEG, PNG (Max 16 MB)</span>
                    </div>
                  )}
                </label>
              </div>

              {error && (
                <div className="error-message">
                  <span className="error-icon">❌</span>
                  {error}
                </div>
              )}

              <div className="button-group">
                <button
                  onClick={handleReset}
                  className="btn btn-secondary"
                  disabled={!selectedFile || loading}
                >
                  Clear
                </button>
                <button
                  onClick={handleAnalyze}
                  className="btn btn-primary"
                  disabled={!selectedFile || loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Analyzing...
                    </>
                  ) : (
                    <>Analyze Image</>
                  )}
                </button>
              </div>
            </div>
          </div>
        ) : (
          /* Results Section */
          <div className="results-section">
            <div className="results-card">
              <h2>Analysis Results</h2>

              {/* Diagnosis Card */}
              <div className="diagnosis-card">
                <div className="diagnosis-icon">
                  {getStageIcon(results.diagnosis)}
                </div>
                <div className="diagnosis-content">
                  <h3>{results.diagnosis}</h3>
                  <p className="stage">{results.stage}</p>
                </div>
              </div>

              {/* Metrics */}
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Confidence</div>
                  <div className="metric-value">
                    {(results.confidence * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Severity Score</div>
                  <div className="metric-value">
                    {results.severity_score}%
                  </div>
                </div>
              </div>

              {/* Severity Indicator */}
              <div className="severity-section">
                <h4>Severity Indicator</h4>
                <div className="severity-bar-container">
                  <div
                    className="severity-bar"
                    style={{
                      width: `${results.severity_score}%`,
                      backgroundColor: getSeverityColor(results.severity_score)
                    }}
                  >
                    <span className="severity-label">{results.severity_score}%</span>
                  </div>
                </div>
                <div className="severity-labels">
                  <span>Healthy</span>
                  <span>Moderate</span>
                  <span>Severe</span>
                </div>
              </div>

              {/* Class Probabilities */}
              <div className="probabilities-section">
                <h4>Class Probabilities</h4>
                <div className="probability-list">
                  {Object.entries(results.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([className, probability]) => (
                      <div key={className} className="probability-item">
                        <div className="probability-header">
                          <span className="class-name">{className}</span>
                          <span className="probability-value">
                            {(probability * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="probability-bar-container">
                          <div
                            className="probability-bar"
                            style={{ width: `${probability * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>

              {/* Recommendations */}
              <div className="recommendations-section">
                <h4>Recommended Actions</h4>
                <ul className="recommendations-list">
                  {results.recommendations.map((rec, index) => (
                    <li key={index}>
                      <span className="check-icon">✓</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Action Button */}
              <button onClick={handleReset} className="btn btn-primary btn-full">
                Analyze Another Image
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="app-footer">
        <p>⚠️ This is an AI-assisted diagnostic tool. Always consult a medical professional.</p>
      </footer>
    </div>
  );
}

export default App;
