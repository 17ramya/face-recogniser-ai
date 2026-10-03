"""
Configuration for the face-recognition backend
==============================================

Every path and tunable lives here so the rest of the code never hard-codes one.
Anything that depends on the machine can be overridden with an environment
variable, which is how the same code runs locally and in a container.

See ``README.md`` (section *Configuration*) for the full table.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Files and folders
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent


def _path(env_name: str, default: Path) -> Path:
    """Return ``env_name`` as a :class:`Path` when set, otherwise ``default``."""
    raw = os.environ.get(env_name)
    return Path(raw).expanduser().resolve() if raw else default


#: Labelled training images, one folder per person: ``dataset/<person>/<n>.jpg``
DATASET_DIR = _path("FACE_DATASET_DIR", BASE_DIR / "dataset")

#: Trained artifacts produced by ``train.py``.
MODELS_DIR = _path("FACE_MODELS_DIR", BASE_DIR / "models")
MODEL_PATH = MODELS_DIR / "lbph_model.yml"
LABELS_PATH = MODELS_DIR / "labels.json"

#: People metadata store (``name``, ``age``, ``department`` ...).
PEOPLE_FILE = _path("FACE_PEOPLE_FILE", BASE_DIR / "people.json")

#: The last image uploaded by the browser is kept here for debugging.
UPLOADS_DIR = _path("FACE_UPLOADS_DIR", BASE_DIR / "uploads")

# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #

#: The React app talks to port 3001 (see ``src/api.js``).
HOST = os.environ.get("FACE_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("FACE_API_PORT", "3001"))

#: Reject uploads larger than this (a 640x480 JPEG is well under 200 KB).
MAX_CONTENT_LENGTH = int(os.environ.get("FACE_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))

# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #

#: LBPH returns a *distance*: 0 is a perfect match, larger is worse.
#: A distance at or below this value is accepted as a real identification.
MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "70"))

#: Every face crop is resized to this square before training/prediction.
FACE_SIZE = (200, 200)

#: Haar-cascade detection tuning.
DETECT_SCALE_FACTOR = float(os.environ.get("FACE_DETECT_SCALE", "1.1"))
DETECT_MIN_NEIGHBORS = int(os.environ.get("FACE_DETECT_MIN_NEIGHBORS", "5"))
DETECT_MIN_SIZE = int(os.environ.get("FACE_DETECT_MIN_SIZE", "60"))

#: When the cascade cannot see a face (a pre-cropped photo, a cartoon, a low
#: resolution webcam frame) fall back to using the whole image. This is what
#: makes the demo usable with tightly cropped pictures; set to ``0`` to demand
#: a detected face.
FULL_IMAGE_FALLBACK = os.environ.get("FACE_FULL_IMAGE_FALLBACK", "1") not in {"0", "false", "no"}

#: LBPH parameters (OpenCV's defaults, spelled out so they can be tuned).
LBPH_RADIUS = int(os.environ.get("FACE_LBPH_RADIUS", "1"))
LBPH_NEIGHBORS = int(os.environ.get("FACE_LBPH_NEIGHBORS", "8"))
LBPH_GRID_X = int(os.environ.get("FACE_LBPH_GRID_X", "8"))
LBPH_GRID_Y = int(os.environ.get("FACE_LBPH_GRID_Y", "8"))

#: Files that count as training images.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".webp"}
