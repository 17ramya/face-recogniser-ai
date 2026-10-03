import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

/**
 * Renders one recognition result.
 *
 * The API always answers with `person: { name, age, department }`, whatever
 * happened, plus `recognized`, `faces`, `detection`, `distance`, `threshold`
 * and a human-readable `message`. The old `people` prop is still accepted so
 * nothing that used this component earlier breaks.
 */

const TONES = {
  match: { badge: 'bg-success', title: 'Recognised' },
  unknown: { badge: 'bg-warning text-dark', title: 'Not in the dataset' },
  noface: { badge: 'bg-secondary', title: 'No face found' },
  notrained: { badge: 'bg-danger', title: 'Model not trained' },
};

function toneOf(payload) {
  if (!payload) {
    return 'unknown';
  }
  if (payload.recognized) {
    return 'match';
  }
  if (payload.detection === 'none') {
    return 'noface';
  }
  if (payload.closest_distance === null || payload.closest_distance === undefined) {
    return 'notrained';
  }
  return 'unknown';
}

function Meta({ payload }) {
  const rows = [
    `Faces detected: ${payload.faces || 0} (${payload.detection || 'n/a'})`,
  ];
  if (typeof payload.distance === 'number') {
    rows.push(`Match distance: ${payload.distance} (threshold ${payload.threshold})`);
  } else if (typeof payload.closest_distance === 'number') {
    rows.push(`Closest distance: ${payload.closest_distance} (threshold ${payload.threshold})`);
  }
  return (
    <ul className="list-unstyled small text-muted mb-0">
      {rows.map((row) => (
        <li key={row}>{row}</li>
      ))}
    </ul>
  );
}

function PersonList({ result, people }) {
  const payload = result || (people ? { person: people, recognized: true } : null);
  if (!payload) {
    return null;
  }
  const person = payload.person || {};
  const tone = TONES[toneOf(payload)];

  return (
    <div className="personlist-container">
      <div className="card shadow-sm">
        <div className="card-header d-flex align-items-center justify-content-between">
          <span className="fw-semibold">Recognition result</span>
          <span className={`badge ${tone.badge}`}>{tone.title}</span>
        </div>
        <div className="card-body">
          <h2 className="h4 mb-1">{person.name || 'Unknown'}</h2>
          <p className="mb-1">
            <strong>Age:</strong> {person.age || '—'}
          </p>
          <p className="mb-3">
            <strong>Department:</strong> {person.department || '—'}
          </p>
          {payload.message && <p className="text-muted small">{payload.message}</p>}
          <Meta payload={payload} />
        </div>
      </div>
    </div>
  );
}

export default PersonList;
