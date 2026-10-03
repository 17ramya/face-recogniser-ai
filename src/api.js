/**
 * The single place that knows where the face-recognition API lives.
 *
 * The default is *this* origin, i.e. relative `/api/...` requests, and that is
 * what both setups need:
 *
 *   - `npm start` - the dev server forwards `/api/*` to the Flask server on
 *     port 3001 thanks to the "proxy" entry in `package.json`.
 *   - Vercel - `vercel.json` rewrites `/api/*` to the internal `backend`
 *     service of the same project, so the deployed site needs no hostname, no
 *     CORS and no service binding (the bundle is static and could not read one
 *     anyway).
 *
 * Set `REACT_APP_API_BASE_URL` in a `.env` file in the project root (Create
 * React App reads it at build/start time) only when the API lives on another
 * machine:
 *
 *     REACT_APP_API_BASE_URL=http://192.168.1.20:3001
 *
 * The ports it must agree with are `backend/config.py` (FACE_API_HOST /
 * FACE_API_PORT).
 */
import axios from 'axios';

export const API_BASE_URL = (process.env.REACT_APP_API_BASE_URL || '').replace(/\/+$/, '');

/** An empty base means "the API is on this origin" - read better in messages. */
const WHERE = API_BASE_URL || 'this site';

export const ENDPOINTS = {
  health: `${API_BASE_URL}/api/health`,
  people: `${API_BASE_URL}/api/people`,
  dataset: `${API_BASE_URL}/api/dataset`,
  storeImage: `${API_BASE_URL}/api/storeimage`,
  reload: `${API_BASE_URL}/api/reload`,
};

/**
 * Flatten whatever the API - or the platform in front of it - answered with
 * into one plain string.
 *
 * A service that crashes is answered by Vercel itself with
 * `{"error": {"code": "500", "message": "A server error has occurred"}}`, so
 * `data.error` is an *object* in the most common failure case. Handing that
 * object back put it where React expects text and blanked the entire page
 * (React error #31, "Objects are not valid as a React child") - an expensive
 * way to learn that the backend is down.
 */
function messageFrom(data) {
  if (data == null) return '';
  if (typeof data === 'string') return data.trim();
  if (typeof data !== 'object') return String(data);
  return (
    messageFrom(data.message) ||
    messageFrom(data.error) ||
    messageFrom(data.code) ||
    JSON.stringify(data)
  );
}

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
        `${WHERE} answered with a web page instead of the API, so the call reached ` +
        'the front end rather than Flask. Check REACT_APP_API_BASE_URL, and on ' +
        'Vercel that the /api/* rewrite in vercel.json still points at the ' +
        'backend service.'
      );
    }
    // `messageFrom` never returns an object, so this is always renderable.
    const text = messageFrom(data);
    if (!text) {
      return `The API answered with status ${error.response.status}.`;
    }
    return error.response.status >= 500
      ? `The API answered with status ${error.response.status}: ${text}`
      : text;
  }
  if (error && error.request) {
    return API_BASE_URL
      ? `Could not reach the face API at ${API_BASE_URL}. ` +
          'Check the backend window: start it with "python app.py" inside the ' +
          'backend folder.'
      : "Could not reach the face API through this site's /api/ routes. " +
          'Start the backend (.\\start-all.ps1), or set REACT_APP_API_BASE_URL ' +
          'to the address of the machine running it.';
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
