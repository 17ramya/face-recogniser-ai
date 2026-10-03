"""
OpenCV face-recognition engine
==============================

A tiny, dependency-light wrapper around the two OpenCV pieces the project
uses:

* **Detection** - the bundled Haar cascade ``haarcascade_frontalface_default.xml``
  finds face rectangles in a grayscale frame.
* **Recognition** - the *contrib* module LBPH (Local Binary Patterns Histograms)
  recognizer learns one histogram per person and predicts by histogram
  distance. It is the classic OpenCV recognizer: no GPU, no dlib, no download.

The engine is deliberately free of Flask and of any knowledge about people
metadata - :mod:`app` supplies a ``resolver`` callable that turns an internal
person key (a dataset folder name) into the ``name``/``age``/``department``
dictionary the React app renders.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

import config

log = logging.getLogger("face.engine")

#: Shown when the camera pointed at the screen but nobody matched.
UNKNOWN_PERSON = {"name": "Unknown", "age": "—", "department": "—"}

Rect = Tuple[int, int, int, int]


def default_resolver(key: str) -> Dict[str, Any]:
    """Fall back to the raw folder name when no metadata is registered."""
    return {"name": key, "age": "—", "department": "—"}


def iter_dataset(dataset_dir: Optional[Path] = None) -> Iterator[Tuple[str, List[Path]]]:
    """Yield ``(person_key, [image paths])`` for every folder of the dataset."""
    root = Path(dataset_dir or config.DATASET_DIR)
    if not root.exists():
        return
    folders = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower())
    for person_dir in folders:
        images = sorted(
            (f for f in person_dir.iterdir()
             if f.is_file() and f.suffix.lower() in config.IMAGE_SUFFIXES),
            key=lambda p: p.name.lower(),
        )
        yield person_dir.name, images


def dataset_summary(dataset_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Describe the dataset: which people exist and how many images each has."""
    root = Path(dataset_dir or config.DATASET_DIR)
    people = [{"name": key, "images": len(images)} for key, images in iter_dataset(root)]
    return {
        "directory": str(root),
        "people": people,
        "people_count": len(people),
        "images": sum(person["images"] for person in people),
    }


class FaceEngine:
    """Detect faces, train an LBPH model from ``dataset/`` and identify crops."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        threshold: Optional[float] = None,
        resolver: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> None:
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.labels_path = Path(labels_path or config.LABELS_PATH)
        self.threshold = float(config.MATCH_THRESHOLD if threshold is None else threshold)
        self.resolver = resolver or default_resolver

        self._recognizer = self.new_recognizer()
        self._labels: List[str] = []
        self._trained = False
        self._mtime: Optional[float] = None
        self._detector: Optional[cv2.CascadeClassifier] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Building blocks
    # ------------------------------------------------------------------ #
    @staticmethod
    def new_recognizer() -> Any:
        """A fresh, untrained LBPH recognizer using the configured parameters."""
        return cv2.face.LBPHFaceRecognizer_create(
            config.LBPH_RADIUS,
            config.LBPH_NEIGHBORS,
            config.LBPH_GRID_X,
            config.LBPH_GRID_Y,
        )

    @property
    def detector(self) -> cv2.CascadeClassifier:
        """The Haar cascade, loaded once, lazily."""
        if self._detector is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(str(cascade_path))
            if detector.empty():
                raise RuntimeError(f"Could not load the Haar cascade from {cascade_path}")
            self._detector = detector
        return self._detector

    def detect_faces(self, gray: np.ndarray) -> List[Rect]:
        """Return face rectangles ``(x, y, w, h)``, largest first."""
        rects = self.detector.detectMultiScale(
            gray,
            scaleFactor=config.DETECT_SCALE_FACTOR,
            minNeighbors=config.DETECT_MIN_NEIGHBORS,
            minSize=(config.DETECT_MIN_SIZE, config.DETECT_MIN_SIZE),
        )
        faces = [tuple(int(v) for v in rect) for rect in rects]
        return sorted(faces, key=lambda r: r[2] * r[3], reverse=True)

    @staticmethod
    def decode_image(data: bytes) -> np.ndarray:
        """Decode raw upload bytes (JPEG/PNG/...) into a BGR image.

        Use :meth:`decode_gray` for recognition: the pixels must match the ones
        ``train()`` read from disk, see the note there.
        """
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded file is not a readable image.")
        return image

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        """Grayscale view of ``image`` (no-op when it already is grayscale)."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    @staticmethod
    def decode_gray(data: bytes) -> np.ndarray:
        """Decode raw upload bytes straight to grayscale - the training path.

        ``train()`` reads the dataset with ``cv2.imread(path, IMREAD_GRAYSCALE)``,
        so uploads are decoded the very same way.  Decoding to colour and then
        calling ``cvtColor(BGR2GRAY)`` is *not* equivalent: the two decoders
        disagree by a grey level or two on a few percent of the pixels of a
        JPEG, which moves the Haar box and rewrites the LBP codes.  LBPH is
        extremely sensitive to that, so an identified photo would come back at
        distance ~70 (rejected) instead of ~1 (accepted).
        """
        buffer = np.frombuffer(data, dtype=np.uint8)
        gray = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("The uploaded file is not a readable image.")
        return gray

    @staticmethod
    def prepare_face(gray: np.ndarray, rect: Optional[Rect] = None) -> np.ndarray:
        """Crop ``rect`` (or the whole frame), resize and histogram-equalise it."""
        if rect is not None:
            x, y, w, h = rect
            crop = gray[y:y + h, x:x + w]
        else:
            crop = gray
        if crop is None or crop.size == 0:
            raise ValueError("The face crop is empty - the rectangle is out of bounds.")
        resized = cv2.resize(crop, config.FACE_SIZE, interpolation=cv2.INTER_AREA)
        return cv2.equalizeHist(resized)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, dataset_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Learn every person in ``dataset/``; returns a report dictionary.

        A photo whose face the cascade cannot see is used whole when
        ``FACE_FULL_IMAGE_FALLBACK`` is on (the default), which is what makes
        pre-cropped portrait photos trainable without any manual boxing.
        """
        root = Path(dataset_dir or config.DATASET_DIR)
        faces: List[np.ndarray] = []
        ids: List[int] = []
        label_names: List[str] = []
        skipped: List[Dict[str, str]] = []
        detection = {"haar": 0, "full-image": 0}

        for key, images in iter_dataset(root):
            if not images:
                skipped.append({"person": key, "reason": "no images"})
                continue
            if key not in label_names:
                label_names.append(key)
            label_id = label_names.index(key)
            for image_path in images:
                gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    skipped.append({"person": key, "image": image_path.name, "reason": "unreadable"})
                    continue
                rects = self.detect_faces(gray)
                if rects:
                    rect, used = rects[0], "haar"
                elif config.FULL_IMAGE_FALLBACK:
                    rect, used = None, "full-image"
                else:
                    skipped.append({"person": key, "image": image_path.name,
                                    "reason": "no face detected"})
                    continue
                faces.append(self.prepare_face(gray, rect))
                ids.append(label_id)
                detection[used] += 1

        if not faces:
            raise ValueError(
                f"No usable training images found in {root}. Add at least one photo per "
                "person under dataset/<person>/ or record them with capture.py."
            )

        recognizer = self.new_recognizer()
        recognizer.train(faces, np.array(ids, dtype=np.int32))
        with self._lock:
            self._recognizer = recognizer
            self._labels = label_names
            self._trained = True

        return {
            "directory": str(root),
            "people": len(label_names),
            "images": len(faces),
            "labels": label_names,
            "detection": detection,
            "skipped": skipped,
        }

    def evaluate(self, dataset_dir: Optional[Path] = None) -> Dict[str, Any]:
        """How well the *training* photos match back - a sanity check, not a test set."""
        root = Path(dataset_dir or config.DATASET_DIR)
        correct = total = 0
        wrong: List[Dict[str, Any]] = []
        distances: List[float] = []
        for key, images in iter_dataset(root):
            for image_path in images:
                gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                rects = self.detect_faces(gray)
                if not rects and not config.FULL_IMAGE_FALLBACK:
                    continue
                rect = rects[0] if rects else None
                prediction = self.classify_face(self.prepare_face(gray, rect))
                total += 1
                if prediction is None:
                    continue
                predicted_key, distance = prediction
                distances.append(distance)
                if predicted_key == key and distance <= self.threshold:
                    correct += 1
                else:
                    wrong.append({"person": key, "image": image_path.name,
                                  "predicted": predicted_key, "distance": round(distance, 2)})
        return {
            "images": total,
            "matched": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "mean_distance": round(sum(distances) / len(distances), 2) if distances else None,
            "misses": wrong[:50],
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self) -> Dict[str, Any]:
        """Persist the trained model and its label map."""
        with self._lock:
            if not self._trained:
                raise RuntimeError("Nothing to save - train the model first.")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self._recognizer.save(str(self.model_path))
            self.labels_path.write_text(json.dumps(self._labels, indent=2), encoding="utf-8")
            self._mtime = self.model_path.stat().st_mtime
            return {"model": str(self.model_path), "labels": str(self.labels_path),
                    "people": len(self._labels)}

    def load(self, force: bool = False) -> bool:
        """Load model + labels from disk. Returns ``True`` when the engine is ready."""
        with self._lock:
            if not (self.model_path.exists() and self.labels_path.exists()):
                self._trained = False
                return False
            mtime = self.model_path.stat().st_mtime
            if self._trained and not force and mtime == self._mtime:
                return True
            recognizer = self.new_recognizer()
            recognizer.read(str(self.model_path))
            labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
            if not isinstance(labels, list):
                raise ValueError(f"{self.labels_path} must contain a JSON list of person keys.")
            self._recognizer = recognizer
            self._labels = [str(label) for label in labels]
            self._trained = True
            self._mtime = mtime
            log.info("loaded model with %d people from %s", len(self._labels), self.model_path)
            return True

    def ensure_fresh(self) -> bool:
        """Re-load when ``train.py`` has rewritten the model. ``True`` if reloaded."""
        if not self.model_path.exists():
            return False
        mtime = self.model_path.stat().st_mtime
        if self._trained and mtime == self._mtime:
            return False
        return self.load(force=True)

    # ------------------------------------------------------------------ #
    # Recognition
    # ------------------------------------------------------------------ #
    @property
    def is_ready(self) -> bool:
        """``True`` once a model has been trained or loaded."""
        return self._trained

    @property
    def labels(self) -> List[str]:
        """The enrolled person keys, in label-id order."""
        return list(self._labels)

    def classify_face(self, gray_face: np.ndarray) -> Optional[Tuple[str, float]]:
        """Predict ``(person_key, distance)`` for an already-cropped face."""
        with self._lock:
            if not self._trained:
                return None
            label_id, distance = self._recognizer.predict(self.prepare_face(gray_face))
            labels = list(self._labels)
        index = int(label_id)
        if 0 <= index < len(labels):
            return labels[index], float(distance)
        return None

    def predict_bytes(self, data: bytes) -> Dict[str, Any]:
        """The whole pipeline: bytes -> detect -> LBPH -> person dictionary."""
        gray = self.decode_gray(data)
        self.ensure_fresh()
        rects = self.detect_faces(gray)
        if rects:
            detection, regions = "haar", list(rects)
        elif config.FULL_IMAGE_FALLBACK:
            detection, regions = "full-image", [None]
        else:
            detection, regions = "none", []

        results: List[Dict[str, Any]] = []
        best: Optional[Tuple[str, float]] = None
        for rect in regions:
            entry: Dict[str, Any] = {"box": list(rect) if rect else None}
            prediction = self.classify_face(self.prepare_face(gray, rect))
            if prediction is None:
                entry.update({"recognized": False, "person": None, "distance": None})
            else:
                key, distance = prediction
                recognized = distance <= self.threshold
                entry.update({"recognized": recognized, "person": key,
                              "distance": round(distance, 2)})
                if recognized and (best is None or distance < best[1]):
                    best = (key, distance)
            results.append(entry)

        distances = [item["distance"] for item in results if item["distance"] is not None]
        payload: Dict[str, Any] = {
            "recognized": best is not None,
            "faces": len(rects),
            "detection": detection,
            "threshold": self.threshold,
            "closest_distance": min(distances) if distances else None,
            "results": results,
        }

        if detection == "none":
            payload["person"] = {"name": "No face detected", "age": "—", "department": "—"}
            payload["message"] = (
                "No face was found in the picture. Move closer to the camera, improve the "
                "lighting, or set FACE_FULL_IMAGE_FALLBACK=1 to use the whole frame."
            )
        elif not self.is_ready:
            payload["person"] = dict(UNKNOWN_PERSON)
            payload["message"] = (
                "The model is not trained yet. Put photos in backend/dataset/<person>/ "
                "and run: python train.py"
            )
        elif best is None:
            payload["person"] = dict(UNKNOWN_PERSON)
            payload["message"] = (
                "A face was found but it did not match anybody in the dataset "
                f"(closest distance {payload['closest_distance']}, threshold {self.threshold})."
            )
        else:
            person = dict(self.resolver(best[0]))
            person.setdefault("name", best[0])
            person["key"] = best[0]
            person["distance"] = round(best[1], 2)
            payload["person"] = person
            payload["match_key"] = best[0]
            payload["distance"] = round(best[1], 2)
            payload["message"] = (
                f"Recognised {person['name']} out of {len(self.labels)} enrolled "
                f"people (distance {person['distance']})."
            )
        return payload
