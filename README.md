# Face Recogniser

Capture a photo in the browser and the person is identified by a face recognizer
trained on your own photographs. **React** front end, **Flask + OpenCV** back
end, running entirely on your machine: no cloud service, no GPU, no `dlib` to
compile and no model weights to download.

This repository originally contained **only the React front end** - the API it
posted to on port `3001` was missing, so pressing *Capture* could never do
anything. This version adds that missing back end, wires the two halves
together and documents how to run the whole thing.

## What it does

| Piece | Detail |
| --- | --- |
| Front end | React 18 (Create React App), `react-webcam`, Bootstrap 5, axios |
| Back end | Flask 3 + `flask-cors`, `opencv-contrib-python`, NumPy |
| Detection | OpenCV Haar cascade `haarcascade_frontalface_default.xml` |
| Recognition | **LBPH** - Local Binary Patterns Histograms, `cv2.face` |
| Training data | `backend/dataset/<person>/` - photos, copied in untouched by `capture.py` |
| Metadata | `backend/people.json` - display name, age, department per person |
| Model | `backend/models/lbph_model.yml` plus `labels.json` |
| Ports | API `3001`, dev server `3000` |
| Checks | 24 pytest checks (`backend/test_app.py`), all on synthetic images |

### The interface

* a live webcam preview with **Capture**, **Retake** and **Save image**;
* a header badge that says whether the API is reachable and whether a model is
  trained, so a broken setup is obvious before you press anything;
* a result card with **name, age, department** and whether the face was
  *recognised*, *not in the dataset* or *not found at all*, plus the LBPH match
  distance, the detection mode and how many faces were seen;
* a footer naming the API it is talking to, the dataset size and the threshold
  in force.

## How it works

```
 browser                    Flask API (port 3001)              artifacts
+--------------+  POST     +------------------------+  cv2.face  +----------+
| react-webcam | /api/     | decode -> Haar detect  | ---------> | LBPH     |
|  "Capture"   | storeimage| crop -> resize 200x200 |  histograms|  .yml    |
| axios FormData| -------> | -> equalizeHist -> LBPH|            +----------+
| result card  | <-------  | distance <= threshold? |   people.json
+--------------+   JSON    +------------------------+
```

1. `react-webcam` grabs a PNG frame and `src/api.js` posts it as a multipart form
   field named `image`.
2. The API decodes it, finds faces with the Haar cascade and, for each face,
   crops, resizes to 200x200 and histogram-equalises it.
3. `cv2.face.LBPHFaceRecognizer` returns the nearest label and its **distance**
   (0 is a perfect match, larger is worse). A distance at or below
   `FACE_MATCH_THRESHOLD` (default `70`) is accepted; anything else is
   *Unknown*.
4. The label is a dataset folder name, which `people.json` turns into
   `{ name, age, department }` for the card.

The frame that is posted is **not** mirrored. The live preview is, because an
image that moves with you is easier to aim, but `getScreenshot()` mirrors along
with it - and every training image comes from `capture.py`/`train.py`, which read
the raw camera frame. Comparing a face with its own reflection is expensive: on a
photo that otherwise matches its training data perfectly, mirroring it moved the
LBPH distance from `0.00` to `49.67` against a threshold of `70`, i.e. most of the
budget spent before the lighting and pose of a real webcam frame are counted. The
page therefore flips the frame back through a canvas, and posts **PNG** rather
than JPEG because that round trip is bit-identical (`0.00`) where a JPEG
re-encode of the same photo cost `26.82`.

If the cascade cannot see a face - a tightly cropped portrait, a cartoon, a dark
webcam frame - the whole image is used instead (`FACE_FULL_IMAGE_FALLBACK`, on by
default). Set `FACE_FULL_IMAGE_FALLBACK=0` if you would rather be told "no face
detected" than be given a guess.

Everything the API answers always contains `person` with `name`, `age` and
`department`, whatever happened, so the front end can always render something
useful.

## Project structure

```
face-recogniser-main/
|-- src/                     React front end
|   |-- index.js             entry point
|   |-- App.js               webcam, capture, status badge, layout
|   |-- api.js               API base URL, the two calls, error wording
|   |-- personlist.js        the recognition result card
|   |-- App.css              layout and responsive rules
|   +-- server.js            legacy leftovers, not imported by index.js
|       sendrequest.js
|       imagecapture.js
|-- public/                  HTML shell, manifest, icons
|-- start-backend.ps1        venv + requirements + API in one command
|-- start-frontend.ps1       npm install if needed + dev server in one command
|-- start-all.ps1            both of the above, each in its own window
|-- start-backend.cmd        cmd.exe twins of the two above, no execution
|-- start-frontend.cmd       policy change needed
|-- .env.example             copy to .env to point the app at another API host
|-- package.json             front-end dependencies and scripts
+-- backend/                 Python API
    |-- app.py               Flask routes and error handlers
    |-- config.py            every path and tunable, overridable by environment
    |-- engine.py            Haar detection + LBPH training/prediction
    |-- people.py            people.json read/write
    |-- train.py             build the model from dataset/
    |-- capture.py           enrol a person (webcam or folder of photos)
    |-- test_app.py          24 pytest checks
    |-- requirements.txt
    |-- people.json          who the model knows about
    |-- dataset/<person>/    training photos     (git-ignored)
    |-- models/              trained artifacts    (git-ignored)
    +-- uploads/             last frame received, `.png`/`.jpg` (git-ignored)
```

## Prerequisites

| Tool | Version used here | Check with |
| --- | --- | --- |
| Python | 3.9 - 3.12 (tested on 3.11.3) | `python --version` |
| pip | ships with Python | `python -m pip --version` |
| Node.js | 18 - 24 (tested on 24.13.1) | `node --version` |
| npm | 9+ (tested on 11.8.0) | `npm --version` |
| Webcam | any camera the browser can reach | - |

Nothing else is installed by hand: `Flask`, `flask-cors`,
`opencv-contrib-python`, `numpy` and `pytest` come from
`backend/requirements.txt`.

## Quick start

From the project folder in PowerShell:

```powershell
cd "C:\Users\RAMYA S\OneDrive\Documents\github\face-recogniser-main"

# 1. front-end dependencies (once)
npm install

# 2. start the API in one window (builds a virtualenv on first run)
.\start-backend.ps1

# 3. start the React app in a second window
.\start-frontend.ps1
```

Then open <http://localhost:3000>, allow the camera and press **Capture**.
From `cmd.exe`, or on a machine whose PowerShell execution policy blocks local
scripts, the `.cmd` twins do exactly the same without changing any setting:

```bat
start-backend.cmd
start-frontend.cmd
```
Straight after a fresh clone the model is untrained, so the page shows an amber
*"No model is trained yet"* banner - train it in the next section.

Once both halves have been installed at least once, one command is enough:

```powershell
.\start-all.ps1
```

macOS / Linux equivalents:

```bash
npm install
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # API on http://127.0.0.1:3001

# in another terminal, from the project root
npm start                # app on http://localhost:3000
```

### The two ports - do not mix them up

| Address | What answers there | What you see |
| --- | --- | --- |
| <http://localhost:3000> | the React app - **this is the website** | the Face Recogniser page, with the webcam preview |
| <http://127.0.0.1:3001> | the Flask API | JSON - or, in a browser, a short page saying "you are looking at the API" |

`src/api.js` points the app at port 3001, so the two must never swap places. If
Create React App finds 3000 occupied it moves to the next free port by itself -
often 3001, straight onto the API - and then the app is talking to itself and
the API's port serves a web page. `start-frontend.ps1` pins the port to 3000 and
refuses to start if it is taken, and the API answers browsers with a page that
says where the app is. If you ever end up in that state anyway, close the dev
server and run `.\start-frontend.ps1` again.

## Enrolling faces

The model learns from folders under `backend/dataset/`. There are three ways to
fill them; pick whichever suits the photos you already have.

**1. Record from the webcam** (30 frames by default):

```powershell
cd backend
.\.venv\Scripts\python.exe capture.py --name "Ramya S" --age 21 --department "Computer Technology"
```

A window opens: `SPACE` pauses or resumes recording, `Q` quits. Frames are saved
only while a face is visible. Move your head slightly between frames - twenty
varied photos beat a hundred identical ones.

**2. Import photographs you already have:**

```powershell
cd backend
.\.venv\Scripts\python.exe capture.py --name "Ramya S" --age 21 --department "Computer Technology" --images "C:\Users\Ramya\Pictures\me"
```

Every readable picture in that folder is copied into the dataset **exactly as it
is** - not cropped. `train.py` then runs the same detect/crop/resize/equalise
pipeline that recognition runs, which is what makes the photo you enrolled come
back at distance ~0. Cropping at enrolment instead would have `train.py` detect a
face *inside* the crop and crop it a second time, comparing the person against a
different region of their own face.

**3. Copy files in yourself.** Create `backend/dataset/<person_key>/` and drop
`*.jpg`, `*.png` or `*.bmp` files in it, then add a matching entry to
`backend/people.json`:

```json
{
  "ramya_s": {
    "name": "Ramya S",
    "age": 21,
    "department": "Computer Technology"
  }
}
```

The key (`ramya_s`) is what `capture.py` produces from the name: lower case,
spaces turned into underscores (see `people.normalise_key`). Lookups are
case-insensitive, and `-`/`_`/spaces are treated as the same character, so
`Ramya S`, `ramya_s` and `ramya-s` all match. If a folder has no entry, the
folder name is shown as the name and age/department appear as `-`.

## Training

```powershell
cd backend
.\.venv\Scripts\python.exe train.py --evaluate
```

```
[face] dataset : 2 people, 24 images in ...\backend\dataset
[face]   - ramya_s: 12 image(s)
[face]   - ravi_k: 12 image(s)
[face] trained  : 2 people on 24 images (haar 18, whole-image 6)
[face] wrote    : ...\backend\models\lbph_model.yml
[face] wrote    : ...\backend\models\labels.json
[face] self-test: 100.0% of 24 training images matched (mean distance 0.41)
[face] done    : the API reloads this model automatically.
```

* `--dataset DIR`, `--models DIR`, `--threshold N` override the defaults.
* `--evaluate` reports how well the *training* photos match back. It is a
  sanity check, not a test score: LBPH is memorising those images.
* `--json` prints the same report as JSON, for scripting.

**You do not have to restart the API.** Every request checks the model's
timestamp, and `POST /api/reload` forces a re-read immediately.

## Running the two halves

| What | Command | Where |
| --- | --- | --- |
| API only | `cd backend` then `.\.venv\Scripts\python.exe app.py` | <http://127.0.0.1:3001> |
| App only | `npm start` | <http://localhost:3000> |
| Both | `.\start-all.ps1` | both of the above |
| Production build | `npm run build` then `npx serve -s build` | static files in `build/` |
| Backend checks | `cd backend`; `.\.venv\Scripts\python.exe -m pytest test_app.py -q` | - |

If `.\.venv\Scripts\python.exe` feels long, activate the environment once:
`.\.venv\Scripts\Activate.ps1`, then plain `python` works.

## API

Base URL is `http://localhost:3001` unless `FACE_API_HOST` / `FACE_API_PORT` say
otherwise. CORS is open, so the page may be served from any port.

| Route | Method | Body | Returns |
| --- | --- | --- | --- |
| `/` | GET | - | JSON banner and the endpoint list; in a browser, a short HTML page pointing at the app |
| `/api/health` | GET | - | `status`, `model_ready`, `labels`, `threshold`, `dataset`, `people_registered` |
| `/api/people` (also `/people`) | GET | - | `count` plus every registered person |
| `/api/dataset` | GET | - | folders and image counts found under `dataset/` |
| `/api/storeimage` | POST | multipart field `image` | the recognition result (below) |
| `/api/reload` | POST | - | re-read the trained model from disk |
| `/name?name=<key>` | GET | - | one person - kept for the old `src/server.js` |

Errors are always JSON, never an HTML page:

| Status | When |
| --- | --- |
| 400 | no `image` field, an empty file, or bytes that are not an image |
| 404 / 405 | unknown route / wrong method |
| 413 | picture larger than `FACE_MAX_UPLOAD_BYTES` (8 MB) |
| 500 | unexpected failure - the server log has the traceback |

```bash
curl -s http://localhost:3001/api/health
curl -s -X POST -F "image=@me.jpg" http://localhost:3001/api/storeimage
curl -s http://localhost:3001/api/people
```

The last frame the browser sent is kept as `backend/uploads/last_capture.png`,
which is the easiest way to see what the camera actually captured. The extension
follows whatever the browser posted (`capture.py`, curl and the test suite send
JPEG; the app itself sends PNG), and only the newest frame is kept.

## Configuration

Backend settings live in `backend/config.py` and every one of them can be
overridden by an environment variable, so nothing has to be edited to change a
port or a threshold.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FACE_API_HOST` | `127.0.0.1` | interface the API binds to |
| `FACE_API_PORT` | `3001` | API port |
| `FACE_MATCH_THRESHOLD` | `70` | LBPH distance accepted as a match |
| `FACE_FULL_IMAGE_FALLBACK` | `1` | use the whole frame when no face is detected |
| `FACE_DATASET_DIR` | `backend/dataset` | training photos |
| `FACE_MODELS_DIR` | `backend/models` | trained artifacts |
| `FACE_PEOPLE_FILE` | `backend/people.json` | people metadata |
| `FACE_UPLOADS_DIR` | `backend/uploads` | last received frame |
| `FACE_DETECT_SCALE` | `1.1` | Haar `scaleFactor` - smaller is slower but finer |
| `FACE_DETECT_MIN_NEIGHBORS` | `5` | Haar `minNeighbors` - higher is stricter |
| `FACE_DETECT_MIN_SIZE` | `60` | smallest face accepted, in pixels |
| `FACE_LBPH_RADIUS` | `1` | LBPH radius |
| `FACE_LBPH_NEIGHBORS` | `8` | LBPH neighbourhood size |
| `FACE_LBPH_GRID_X` / `_Y` | `8` / `8` | LBPH grid - raise for finer histograms |
| `FACE_MAX_UPLOAD_BYTES` | `8388608` | upload cap (8 MB) |
| `FACE_DEBUG` | `0` | `1` turns on the Flask debugger (local use only) |

The front end reads one variable from `.env` (copy `.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `REACT_APP_API_BASE_URL` | `http://localhost:3001` | where the API is |

Example - a second camera on another port, with a stricter threshold:

```powershell
$env:FACE_API_PORT = '3002'
$env:FACE_MATCH_THRESHOLD = '55'
.\start-backend.ps1 -SkipInstall
```

## Getting good results

LBPH is a template matcher, not a deep network, so the photos matter more than
the settings:

* **10-30 photos per person**, taken the way you will actually be seen: same
  glasses, same lighting, roughly the same distance.
* **Variety beats volume.** Slightly different angles, expressions and
  backgrounds make the histogram robust; thirty identical frames do not. Copying
  the same picture in twice - a folder imported twice, or the same file under two
  names - adds nothing, and the dataset count calling it "2 images" is a
  comfortable lie worth checking.
* **Enrol with the camera you will be recognised by.** `capture.py --name "Your
  Name" --count 20` writes whole frames from that webcam, so the training set
  shares its lighting, distance and lens with the frames the app posts. A studio
  photo on its own can easily land past the threshold on a live frame, even
  though `train.py --evaluate` reports a perfect self-test on it.
* **One face per photo** where you can. When several faces are found the largest
  one is the one that is reported.
* **Even light and a plain background** help the cascade and the recognizer
  alike.

Reading the numbers on the result card:

| Symptom | Meaning | What to change |
| --- | --- | --- |
| distance below about 40 | a solid match | nothing |
| distance 40-70 | a match, but shaky | add more photos of that person |
| distance just above 70 | probably the right person, rejected | raise `FACE_MATCH_THRESHOLD`, e.g. `80` |
| everyone matches one person | threshold too loose, or one person dominates the dataset | lower the threshold, balance the dataset |
| `detection: none` | the cascade saw no face | better light, face the camera; keep `FACE_FULL_IMAGE_FALLBACK=1` |
| `detection: full-image` every time | the cascade never fires on your photos | enrol photos that show the whole face - the fallback is doing all the work |

For one run: `$env:FACE_MATCH_THRESHOLD = '80'`. Permanently: `backend/config.py`.

## Verification

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest test_app.py -q
# ........................
# 24 passed
```

The 24 checks run on synthetic images generated with NumPy, so no webcam, no
dataset and no network are involved:

| Group | What is checked |
| --- | --- |
| Store | key normalisation, add/update round trip, unrelated fields survive, unknown people |
| Engine | dataset walk, empty dataset rejected, training recognises its own images, model save and reload, `ensure_fresh`, evaluation, unreadable bytes rejected |
| HTTP | `/`, `/api/health`, `/api/people`, `/api/dataset`, `/name`, `/api/reload`, JSON 404 and 405 |
| Contract | `/api/storeimage` always answers with `person.name`/`age`/`department`; a known face returns the right name, age and department; bad requests get a JSON 400; the debug copy of the frame keeps the extension it arrived with |

The front end is checked by the build itself, with warnings treated as errors:

```powershell
$env:CI = 'true'
npm run build
# Compiled successfully.
```

### End-to-end check over real HTTP

```powershell
curl -s http://localhost:3001/api/health
curl -s -X POST -F "image=@photo.jpg" http://localhost:3001/api/storeimage
```

With a two-person model the second call answers, for example:

```json
{"recognized": true, "detection": "full-image", "distance": 43.89, "faces": 0,
 "match_key": "alice",
 "person": {"name": "Alice Rao", "age": 20, "department": "Computer Technology"},
 "message": "Recognised Alice Rao out of 2 enrolled people (distance 43.89).",
 "stored": "...\\backend\\uploads\\last_capture.jpg"}
```

## Troubleshooting

| Problem | Cause and fix |
| --- | --- |
| The page shows **"API offline"** | The API is not running. Start it (`.\start-backend.ps1`) and reload. Hover the badge for the real error. |
| Opening `localhost:3001` shows only JSON, or a page saying *"you are looking at the API"* | That is the API - it is supposed to answer like that. The website is on <http://localhost:3000>. If nothing answers there, run `.\start-frontend.ps1`. |
| The app loads but every capture fails with *"answered with a web page instead of the API"* | The dev server has taken port 3001, so the app is calling itself. Close that terminal and run `.\start-frontend.ps1` - it now pins the port to 3000 and tells you if something else holds it. |
| **"No model is trained yet"** | `backend/models/lbph_model.yml` is missing. Add photos and run `python train.py`. |
| `Failed to compile. [eslint] ...` | `npm run build` treats warnings as errors only when `CI=true`; fix the reported file or build without `CI`. |
| `ModuleNotFoundError: No module named 'cv2'` | Wrong interpreter. Use `backend\.venv\Scripts\python.exe`, or activate the venv first. |
| `module 'cv2' has no attribute 'face'` | `opencv-python` is installed instead of `opencv-contrib-python`. Re-run `pip install -r requirements.txt`; only the contrib build has `cv2.face`. |
| Nothing happens when I press Capture | The browser blocked the camera. Click the camera icon in the address bar, allow access, then reload. A camera also needs `localhost` or HTTPS. |
| `Could not reach the face API at http://localhost:3001` | The API died or the port is wrong. Check the backend window; if you move the port, change `FACE_API_PORT` and `REACT_APP_API_BASE_URL` together. |
| `Port 3001 is already in use` | Find the owner with `Get-NetTCPConnection -LocalPort 3001`, or start the API with `FACE_API_PORT=3002`. |
| Everyone is recognised as the same person | Too few photos, one person dominating the dataset, or the threshold is too loose. Add photos of the others and lower `FACE_MATCH_THRESHOLD`. |
| Recognition is always "Unknown" | The threshold is too strict for your lighting, or the training photos look nothing like the live camera - a single studio photo of you, or the same photo stored twice under two names, is the usual reason. Enrol a round of frames with `capture.py --name "Your Name" --count 20`, retrain, and see *Getting good results*. |
| `capture.py` cannot open the camera | Another app (Teams, Zoom) is holding it. Close it, or pass `--camera 1`. |
| `train.py` says "No usable training images" | `backend/dataset/` has no folder containing images. Check that `backend/dataset/<person>/*.jpg` exists. |
| The age or department is wrong | Edit `backend/people.json` - training does not rebuild it. |
| Deprecation or Browserslist warnings on every command | Harmless. `npx update-browserslist-db@latest` refreshes the browser list if the nag bothers you. |

## Limitations - please read

* **This is an educational demo, not a security control.** LBPH on a handful of
  photos can be fooled by a printed picture of an enrolled person, and its
  numbers are distances, not calibrated probabilities. Do not use it to unlock
  anything that matters.
* **It only knows the people you showed it.** There is no "none of the above"
  class: an unenrolled face is either rejected by the distance threshold or
  matched to the nearest enrolled person.
* **It degrades quickly with headcount.** Results stay good up to roughly a
  dozen people with a dozen photos each; beyond that LBPH starts confusing
  people and training gets slow.
* **Pose and light matter.** A profile, a hat, or a hard side-light produces a
  very different histogram from the frontal, evenly lit photo you enrolled.
* **Everything happens locally.** Photos never leave the machine, but the last
  frame is written to `backend/uploads/` and the enrolled photos stay in
  `backend/dataset/` - both git-ignored, both yours to delete.
* **The Flask server is a development server.** For anything beyond your own
  machine, put it behind a real WSGI server (waitress, gunicorn) and keep the
  API bound to `127.0.0.1`.

## Legacy files

`src/server.js`, `src/sendrequest.js` and `src/imagecapture.js` came with the
original upload and are **not imported anywhere** - `src/index.js` renders
`src/App.js` only, so they are not in the bundle. They are kept so nothing that
was there before is lost. Note that `src/server.js` is a React component despite
its name, so running it with `node` will not work; the API does still answer the
two shapes they used (`GET /name` and a JSON `POST /api/storeimage`) so they
would work if you ever wired them back in.

## Credits and licence

* Front end built with [Create React App](https://create-react-app.dev/),
  [react-webcam](https://github.com/mozmorris/react-webcam) (MIT),
  [Bootstrap](https://getbootstrap.com/) (MIT),
  [axios](https://axios-http.com/) (MIT) and
  [file-saver](https://github.com/eligrey/FileSaver.js/) (MIT).
* Recognition uses [OpenCV](https://opencv.org/) (Apache 2.0) - the Haar cascade
  and the `cv2.face` LBPH recognizer - served by
  [Flask](https://flask.palletsprojects.com/) (BSD).
* Data: whatever photos you enrol. Nothing is downloaded and no external face
  dataset is bundled.

Use it, change it, learn from it. If you deploy it where other people can reach
it, say on the page what it is - and what it is not.

