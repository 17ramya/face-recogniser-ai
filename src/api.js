/**
 * The single place that knows where the face-recognition API lives.
 *
 * Point the app at another machine by creating a `.env` file in the project
 * root (Create React App reads it at build/start time):
 *
 *     REACT_APP_API_BASE_URL=http://192.168.1.20:3001
 *
 * The default matches `backend/config.py` (FACE_API_HOST / FACE_API_PORT).
 */
import axios from 'axios';

export const API_BASE_URL = (
  process.env.REACT_APP_API_BASE_URL || 'http://localhost:3001'
).replace(/\/+$/, '');

export const ENDPOINTS = {
  health: `${API_BASE_URL}/api/health`,
  people: `${API_BASE_URL}/api/people`,
  dataset: `${API_BASE_URL}/api/dataset`,
  storeImage: `${API_BASE_URL}/api/storeimage`,
  reload: `${API_BASE_URL}/api/reload`,
};

/** Turn an axios error into one sentence a human can act on. */
export function describeError(error) {
  if (error && error.response) {
    const data = error.response.data;
    // A web page where JSON was expected almost always means the API URL is
    // pointing at the React dev server instead of the backend - which happens
    // when the dev server has drifted off port 3000 onto the API's port. See
    // the "Two ports" section of README.md.
    if (typeof data === 'string' && /^\s*<(?:!doctype|html)/i.test(data)) {
      return (
        `${API_BASE_URL} answered with a web page instead of the API. ` +
        'Check the ports: the app runs on 3000 and the API on 3001 ' +
        '(REACT_APP_API_BASE_URL).'
      );
    }
    const message = data && typeof data === 'object' ? data.error || data.message : data;
    return message || `The API answered with status ${error.response.status}.`;
  }
  if (error && error.request) {
    return (
      `Could not reach the face API at ${API_BASE_URL}. ` +
      'Start it with "python app.py" inside the backend folder.'
    );
  }
  return (error && error.message) || 'Something unexpected went wrong.';
}

/** GET /api/health - used for the little status badge in the header. */
export async function fetchHealth() {
  const { data } = await axios.get(ENDPOINTS.health, { timeout: 5000 });
  return data;
}

/** POST /api/storeimage - send one captured frame, get the person back. */
export async function recognise(blob, filename = 'capturedImage.jpg') {
  const form = new FormData();
  form.append('image', blob, filename);
  const { data } = await axios.post(ENDPOINTS.storeImage, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 20000,
  });
  return data;
}

const api = { API_BASE_URL, ENDPOINTS, fetchHealth, recognise, describeError };

export default api;
