"""
Face-recognition API
====================

The Flask service the React front end (``src/App.js``) posts webcam frames to.

Run locally::

    python app.py                     # -> http://127.0.0.1:3001

Endpoints
---------
=======================  ======  ==================================================
Route                    Method  Purpose
=======================  ======  ==================================================
``/``                     GET     service banner, endpoint list
``/api/health``           GET     is the API up, is a model trained, dataset summary
``/api/people``           GET     everybody registered in ``people.json``
``/api/dataset``          GET     folders and image counts found under ``dataset/``
``/api/storeimage``       POST    multipart field ``image`` -> recognition result
``/api/reload``           POST    re-read the trained model from disk
``/name``                 GET     legacy helper used by the old ``src/server.js``
=======================  ======  ==================================================

The response of ``/api/storeimage`` always carries ``person`` with the keys
``name``, ``age`` and ``department`` so the front end can render it whatever
the outcome, plus ``recognized``, ``faces`` and ``message`` for diagnostics.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
from markupsafe import escape

import config
from engine import FaceEngine, dataset_summary
from people import PeopleStore

logging.basicConfig(level=logging.INFO, format="[face] %(levelname)s %(message)s")
log = logging.getLogger("face.app")

# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
store = PeopleStore()
engine = FaceEngine(resolver=store.details_for)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# The React dev server runs on http://localhost:3000, so the browser needs CORS.
CORS(app)


def _status() -> Dict[str, Any]:
    """The shared health block: model state plus dataset state."""
    return {
        "model_ready": engine.is_ready,
        "labels": engine.labels,
        "threshold": engine.threshold,
        "model_path": str(engine.model_path),
        "dataset": dataset_summary(),
    }


def _remember(data: bytes, filename: str = "") -> str:
    """Keep the last uploaded frame on disk so it can be inspected.

    The suffix follows the upload - the browser posts PNG, because a mirrored
    webcam screenshot is flipped back to the camera's frame losslessly in the
    page - so the file on disk is really the format its name claims. Any older
    ``last_capture.*`` frame is removed, leaving exactly one to look at.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in config.IMAGE_SUFFIXES:
        suffix = ".jpg"
    target = config.UPLOADS_DIR / f"last_capture{suffix}"
    try:
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        for stale in config.UPLOADS_DIR.glob("last_capture.*"):
            if stale != target:
                stale.unlink(missing_ok=True)
        target.write_bytes(data)
        return str(target)
    except OSError as exc:  # pragma: no cover - defensive
        log.warning("could not store the upload: %s", exc)
        return ""


#: Where the React app lives. Used only to point people in the right direction
#: when they open the API's own port by mistake.
APP_URL = "http://localhost:3000"

#: ``(label, href, what it does)``. The JSON answer lists the labels; the
#: landing page turns the ones you can open with a GET into links.
ENDPOINTS: Tuple[Tuple[str, str, str], ...] = (
    ("GET /api/health", "/api/health", "is the API up, is a model trained"),
    ("GET /api/people", "/api/people", "everybody registered in people.json"),
    ("GET /api/dataset", "/api/dataset", "folders and image counts found under dataset/"),
    ("POST /api/storeimage", "", "send a webcam frame and get the person back"),
    ("POST /api/reload", "", "re-read a freshly trained model without a restart"),
    ("GET /name?name=<key>", "/name", "legacy helper used by the old src/server.js"),
)

#: A ``string.Template`` so the CSS braces below stay literal.
LANDING_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>face-recogniser API</title>
<style>
  body { margin: 0; background: #f5f6f8; color: #1d2330;
         font: 16px/1.55 system-ui, "Segoe UI", Roboto, sans-serif; }
  main { max-width: 44rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
  h1 { margin: 0 0 .35rem; font-size: 1.45rem; }
  h2 { margin: 2rem 0 .5rem; font-size: .8rem; text-transform: uppercase;
       letter-spacing: .07em; color: #5b6474; }
  .card { background: #fff; border: 1px solid #e2e5ea; border-radius: .5rem;
          padding: 1rem 1.25rem; }
  .cta { margin: 1rem 0 0; padding: .75rem 1rem; background: #eaf2ff;
         border-left: 4px solid #0d6efd; border-radius: .25rem; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #eef0f3;
           vertical-align: top; }
  tr:last-child th, tr:last-child td { border-bottom: 0; }
  th { width: 11rem; font-weight: 600; color: #5b6474; }
  code { background: #f1f3f6; padding: .1rem .35rem; border-radius: .25rem;
         font-size: .875rem; }
  ul { margin: .25rem 0; padding-left: 1.2rem; }
  li { margin: .35rem 0; }
  a { color: #0b5ed7; }
  .hint { color: #5b6474; font-size: .875rem; }
</style>
</head>
<body>
<main>
  <h1>face-recogniser API</h1>
  <p>You are looking at the <strong>API</strong>, not the app. It only speaks
  JSON, so there is no page to browse here - this notice is the whole site.</p>
  <p class="cta">The web app is on <a href="$app_url">$app_url</a>. Open that
  instead; if nothing answers there, start it with
  <code>.\\start-frontend.ps1</code>. The API itself is started by
  <code>.\\start-backend.ps1</code>.</p>

  <h2>Status</h2>
  <div class="card">
    <table>$status_rows</table>
  </div>

  <h2>Endpoints</h2>
  <div class="card">
    <ul>$route_items</ul>
  </div>

  <p class="hint">This page appears because your browser asked for HTML.
  Scripts still get JSON from the same URL - for example
  <code>curl $api_url/api/health</code>.</p>
</main>
</body>
</html>
"""
)


def _api_url() -> str:
    """The URL a human should type to reach this API."""
    host = "127.0.0.1" if config.HOST in {"0.0.0.0", "::"} else config.HOST
    return f"http://{host}:{config.PORT}"


def _wants_html() -> bool:
    """``True`` when the caller looks like a browser rather than a script.

    ``curl`` and the test client send ``*/*`` (or nothing) and keep getting
    JSON; Chrome, Edge and Firefox send ``text/html`` and get the landing page.
    """
    accept = request.headers.get("Accept", "")
    return "text/html" in accept and "application/json" not in accept


def _landing_page() -> str:
    """A short HTML page for anybody who opens the API's port in a browser."""
    dataset = dataset_summary()
    rows = (
        ("API", f"up on port {config.PORT}"),
        ("model trained", "yes" if engine.is_ready else "not yet - run: python train.py"),
        ("people enrolled", str(len(store.all()))),
        ("dataset", f"{dataset['people_count']} people, {dataset['images']} images"),
        ("match threshold", f"{engine.threshold:.0f}"),
        ("model file", str(engine.model_path)),
    )
    status_rows = "\n      ".join(
        f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>" for name, value in rows
    )
    route_items = "\n      ".join(
        f'<li><a href="{escape(href)}"><code>{escape(label)}</code></a> &ndash; {escape(what)}</li>'
        if href
        else f"<li><code>{escape(label)}</code> &ndash; {escape(what)}</li>"
        for label, href, what in ENDPOINTS
    )
    return LANDING_TEMPLATE.substitute(
        app_url=escape(APP_URL),
        api_url=escape(_api_url()),
        status_rows=status_rows,
        route_items=route_items,
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index() -> Any:
    """Service banner: JSON for scripts, a short HTML notice for browsers.

    Opening <http://localhost:3001> by mistake used to look like a broken
    website - it is the API, and the app it belongs to lives on port 3000.
    """
    engine.ensure_fresh()
    if _wants_html():
        return _landing_page()
    return jsonify(
        {
            "service": "face-recogniser API",
            "version": "1.0.0",
            "people_registered": len(store.all()),
            "model_ready": engine.is_ready,
            "app_url": APP_URL,
            "endpoints": [label for label, _href, _what in ENDPOINTS],
        }
    )


@app.route("/api/health")
def health() -> Any:
    engine.ensure_fresh()
    payload = _status()
    payload.update(
        {
            "status": "ok",
            "people_registered": len(store.all()),
            "full_image_fallback": config.FULL_IMAGE_FALLBACK,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    return jsonify(payload)


@app.route("/api/people")
@app.route("/people")
def list_people() -> Any:
    people = store.all()
    return jsonify({"count": len(people), "people": people})


@app.route("/api/dataset")
def dataset() -> Any:
    return jsonify(dataset_summary())


@app.route("/name")
def legacy_name() -> Any:
    """Compatibility with the old ``src/server.js`` component."""
    key = request.args.get("name")
    if key:
        person = store.get(key)
        if person is None:
            return jsonify({"error": f"No person matching {key!r} is registered."}), 404
        return jsonify(person)
    people = store.all()
    if not people:
        return jsonify({"error": "people.json has no entries yet."}), 404
    return jsonify(people[0])


@app.route("/api/reload", methods=["POST", "GET"])
def reload_model() -> Any:
    """Pick up a model that ``train.py`` has just rewritten, without a restart."""
    engine.load(force=True)
    return jsonify({"model_ready": engine.is_ready, "labels": engine.labels})


@app.route("/api/storeimage", methods=["POST"])
def storeimage() -> Tuple[Any, int] | Any:
    """Receive a webcam frame and answer with the recognised person."""
    # The old src/sendrequest.js posted JSON instead of a picture - stay polite.
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        person = store.get(payload.get("name", ""))
        return jsonify(
            {
                "person": person,
                "received": payload,
                "message": "No image was sent. This endpoint expects a multipart field named 'image'.",
            }
        )

    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "Send the picture as a multipart form field named 'image'."}), 400

    data = upload.read()
    if not data:
        return jsonify({"error": "The uploaded file is empty."}), 400

    stored = _remember(data, upload.filename)
    try:
        result = engine.predict_bytes(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("recognition failed")
        return jsonify({"error": f"Recognition failed: {exc}"}), 500

    result["stored"] = stored
    log.info(
        "storeimage -> recognized=%s faces=%s person=%s",
        result.get("recognized"),
        result.get("faces"),
        (result.get("person") or {}).get("name"),
    )
    return jsonify(result)


# --------------------------------------------------------------------------- #
# Errors - always JSON, so the front end can show a real message
# --------------------------------------------------------------------------- #
@app.errorhandler(404)
def not_found(_error: Any) -> Any:
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def method_not_allowed(_error: Any) -> Any:
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(413)
def payload_too_large(_error: Any) -> Any:
    limit = config.MAX_CONTENT_LENGTH // 1024
    return jsonify({"error": f"The picture is larger than the {limit} KB limit."}), 413


@app.errorhandler(500)
def server_error(_error: Any) -> Any:
    return jsonify({"error": "Internal server error"}), 500


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    if engine.load():
        log.info("trained model found: %d people %s", len(engine.labels), engine.labels)
    else:
        log.warning("no trained model yet - add photos to dataset/ then run: python train.py")
    summary = dataset_summary()
    log.info("dataset: %d people, %d images (%s)",
             summary["people_count"], summary["images"], summary["directory"])
    log.info("listening on http://%s:%s", config.HOST, config.PORT)
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=os.environ.get("FACE_DEBUG", "0") in {"1", "true", "yes"},
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
