import React, { useCallback, useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import fileSaver from 'file-saver';
import './App.css';
import PersonList from './personlist';
import { API_BASE_URL, describeError, fetchHealth, recognise } from './api';

/** A webcam screenshot is a data URL; the API wants a real file. */
function dataURLToBlob(dataURL) {
  const byteString = atob(dataURL.split(',')[1]);
  const mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
  const buffer = new ArrayBuffer(byteString.length);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < byteString.length; i += 1) {
    bytes[i] = byteString.charCodeAt(i);
  }
  return new Blob([buffer], { type: mimeString });
}

/**
 * Flip a webcam screenshot back to the frame the camera really saw.
 *
 * The `mirrored` prop flips the preview *and* `getScreenshot()`, so the frame
 * that arrives here is a mirror image of reality. Every training image comes
 * from `capture.py`/`train.py`, which read the raw, unmirrored camera frame, so
 * posting a mirrored frame asks LBPH to compare a face with its own reflection.
 * Measured on a dataset photo that matches its training data perfectly: 0.00
 * unmirrored against 49.67 mirrored, with the threshold at 70 - most of the
 * budget gone before lighting and pose are counted.
 *
 * PNG rather than JPEG for the flip: that round trip is bit-identical (0.00 on
 * the same photo), where re-encoding it as JPEG q75 cost 26.82.
 */
function unmirrorDataURL(dataURL) {
  return new Promise((resolve, reject) => {
    const frame = new Image();
    frame.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = frame.naturalWidth;
      canvas.height = frame.naturalHeight;
      const context = canvas.getContext('2d');
      context.translate(canvas.width, 0);
      context.scale(-1, 1);
      context.drawImage(frame, 0, 0);
      resolve(canvas.toDataURL('image/png'));
    };
    frame.onerror = () => reject(new Error('Could not read the captured frame.'));
    frame.src = dataURL;
  });
}

const videoConstraints = {
  width: 480,
  height: 360,
  facingMode: 'user',
};

function App() {
  const webcamRef = useRef(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await fetchHealth());
    } catch (err) {
      setHealth({ status: 'offline', error: describeError(err) });
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const timer = setInterval(refreshHealth, 15000);
    return () => clearInterval(timer);
  }, [refreshHealth]);

  const capture = async () => {
    if (busy) return;
    const webcam = webcamRef.current;
    const shot = webcam && webcam.getScreenshot();
    if (!shot) {
      setError('The camera is not ready yet - give it a second and try again.');
      return;
    }
    setImageSrc(shot);
    setResult(null);
    setError('');
    setBusy(true);
    try {
      // Post the unmirrored frame, then show that same frame, so what you are
      // looking at afterwards is exactly what the API was given.
      const posted = await unmirrorDataURL(shot);
      setImageSrc(posted);
      setResult(await recognise(dataURLToBlob(posted), 'capturedImage.png'));
      refreshHealth();
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const saveImageToFile = () => {
    if (imageSrc) {
      fileSaver.saveAs(dataURLToBlob(imageSrc), 'capturedImage.jpg');
    }
  };

  const online = health && health.status === 'ok';
  const trained = Boolean(health && health.model_ready);

  return (
    <div className="app-shell">
      <header className="app-header bg-primary text-white py-3">
        <div className="container">
          <div className="d-flex flex-wrap align-items-center justify-content-between">
            <div>
              <h1 className="h4 mb-0">Face Recogniser</h1>
              <p className="mb-0 small opacity-75">
                Capture a photo and the API looks the face up in the trained dataset.
              </p>
            </div>
            <span
              className={`badge p-2 ${
                online ? (trained ? 'bg-success' : 'bg-warning text-dark') : 'bg-danger'
              }`}
              title={(health && health.error) || API_BASE_URL}
            >
              {online
                ? trained
                  ? `API online - ${health.labels.length} people trained`
                  : 'API online - no model trained yet'
                : 'API offline'}
            </span>
          </div>
        </div>
      </header>

      <main className="container py-4">
        {!online && (
          <div className="alert alert-danger" role="alert">
            <strong>The face API is not reachable.</strong>{' '}
            <span>
              Start it with <code>python app.py</code> inside the <code>backend</code> folder,
              then reload this page.
            </span>
            {health && health.error && <div className="small mt-1">{health.error}</div>}
          </div>
        )}

        {online && !trained && (
          <div className="alert alert-warning" role="alert">
            <strong>No model is trained yet.</strong>{' '}
            <span>
              Record faces with <code>python capture.py --name &quot;Your Name&quot;</code> and
              run <code>python train.py</code> in the <code>backend</code> folder.
            </span>
          </div>
        )}

        {error && (
          <div className="alert alert-danger" role="alert">
            {error}
          </div>
        )}

        <div className="webcam-personlist-container">
          <div className="webcam-panel">
            {imageSrc ? (
              <img src={imageSrc} alt="Captured frame" className="webcam-frame" />
            ) : (
              <Webcam
                audio={false}
                ref={webcamRef}
                mirrored
                screenshotFormat="image/png"
                videoConstraints={videoConstraints}
                className="webcam-frame"
              />
            )}
            <div className="d-flex gap-2 mt-3 flex-wrap">
              <button type="button" className="btn btn-primary" onClick={capture} disabled={busy}>
                {busy ? 'Recognising...' : 'Capture'}
              </button>
              <button
                type="button"
                className="btn btn-outline-secondary"
                onClick={() => {
                  setImageSrc(null);
                  setResult(null);
                  setError('');
                }}
                disabled={busy || !imageSrc}
              >
                Retake
              </button>
              <button
                type="button"
                className="btn btn-outline-secondary"
                onClick={saveImageToFile}
                disabled={!imageSrc}
              >
                Save image
              </button>
            </div>
            <p className="text-muted small mt-2 mb-0">
              {imageSrc
                ? 'The exact frame the API was given. The live preview is mirrored like a ' +
                  'mirror; the frame that is posted is not. Press Retake for a live camera.'
                : 'Allow camera access, look at the lens and press Capture.'}
            </p>
          </div>

          {result && <PersonList result={result} />}
        </div>
      </main>

      <footer className="container pb-4">
        <hr />
        <p className="text-muted small mb-1">
          API: <code>{API_BASE_URL}</code>
          {health && health.dataset
            ? ` - dataset: ${health.dataset.people_count} people, ${health.dataset.images} images`
            : ''}
          {health && health.threshold ? ` - match threshold ${health.threshold}` : ''}
        </p>
        <p className="text-muted small mb-0">
          Educational demo: Local Binary Patterns Histograms on a handful of photos - not a
          security product.
        </p>
      </footer>
    </div>
  );
}

export default App;
