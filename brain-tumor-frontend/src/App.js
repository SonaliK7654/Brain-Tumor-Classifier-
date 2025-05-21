import React, { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setPrediction(null);
    setConfidence(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!file) {
      setError('Please select an image first');
      return;
    }
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/predict/', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Prediction failed');
      }
      const data = await res.json();
      setPrediction(data.predicted_class);
      setConfidence((data.confidence * 100).toFixed(2));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 500, margin: 'auto', padding: 20 }}>
      <h1>Brain Tumor Classifier</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} />
      <br /><br />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>

      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {prediction !== null && (
        <div style={{ marginTop: 20 }}>
          <h3>Prediction Result</h3>
          <p><strong>Class:</strong> {prediction}</p>
          <p><strong>Confidence:</strong> {confidence} %</p>
        </div>
      )}
    </div>
  );
}

export default App;
