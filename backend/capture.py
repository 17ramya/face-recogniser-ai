"""
Enrol a face from the webcam or from a folder of photos
=======================================================

Creates ``dataset/<person_key>/``, copies photographs (or grayscale camera
frames) into it untouched and registers the person in ``people.json``.
Afterwards run ``train.py``, which finds the face in each picture.

Nothing is cropped here on purpose: ``train.py`` runs the very same
detect/crop/resize/equalise pipeline that recognition runs, so a photo enrolled
with this script is recognised at distance ~0 instead of being cropped twice.

Usage::

    # record 30 frames from the default camera (SPACE pauses, Q quits)
    python capture.py --name "Ramya S" --age 21 --department "Computer Technology"

    # import existing photographs instead of using the camera
    python capture.py --name "Ramya S" --images "C:/photos/ramya"

    # a different camera, or more frames
    python capture.py --name "Ravi" --camera 1 --count 50

The person key is the dataset folder name: ``Ramya S`` becomes ``ramya_s``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import cv2

import config
from engine import FaceEngine
from people import PeopleStore, normalise_key

WINDOW = "enrol - SPACE: pause/resume, Q: quit"


def slugify(name: str) -> str:
    """Dataset folder name for a person: ``Ramya S`` -> ``ramya_s``."""
    return normalise_key(name).replace(" ", "_") or "person"


def existing_images(target: Path) -> int:
    """How many images already sit in ``target``; new files carry on from there."""
    if not target.is_dir():
        return 0
    return sum(1 for path in target.iterdir()
               if path.is_file() and path.suffix.lower() in config.IMAGE_SUFFIXES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capture.py",
        description="Record training images for one person from a webcam or a folder.",
    )
    parser.add_argument("--name", required=True, help="display name, e.g. \"Ramya S\"")
    parser.add_argument("--age", help="age to show in the UI")
    parser.add_argument("--department", help="department to show in the UI")
    parser.add_argument("--count", type=int, default=30, help="frames to record (default: 30)")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--images", type=Path,
                        help="import every picture of this folder instead of using a camera")
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR,
                        help="dataset root (default: backend/dataset)")
    parser.add_argument("--no-register", action="store_true",
                        help="only write images, do not touch people.json")
    return parser


def import_folder(engine: FaceEngine, source: Path, target: Path) -> int:
    """Copy every readable picture of ``source`` into ``target``, byte for byte."""
    if not source.is_dir():
        raise SystemExit(f"{source} is not a folder.")
    saved = 0
    offset = existing_images(target)
    pictures = sorted(p for p in source.iterdir()
                      if p.is_file() and p.suffix.lower() in config.IMAGE_SUFFIXES)
    if not pictures:
        raise SystemExit(f"No images ({', '.join(sorted(config.IMAGE_SUFFIXES))}) found in {source}.")
    for picture in pictures:
        gray = cv2.imread(str(picture), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[face] skipped {picture.name}: unreadable")
            continue
        if not engine.detect_faces(gray):
            if not config.FULL_IMAGE_FALLBACK:
                print(f"[face] skipped {picture.name}: no face detected")
                continue
            print(f"[face] note: no face found in {picture.name}; "
                  "train.py will use the whole picture for it")
        saved += 1
        shutil.copyfile(picture, target / f"photo_{offset + saved:03d}{picture.suffix.lower()}")
    return saved


def capture_from_camera(engine: FaceEngine, target: Path, index: int, count: int) -> int:
    """Record ``count`` frames that contain a face from a live camera."""
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        print(f"[face] error: could not open camera {index}; "
              f"use --images DIR to import photographs instead.", file=sys.stderr)
        return 0
    saved = 0
    offset = existing_images(target)
    recording = True
    print(f"[face] {WINDOW}")
    try:
        while saved < count:
            ok, frame = capture.read()
            if not ok:
                print("[face] error: the camera returned no frame.", file=sys.stderr)
                break
            gray = engine.to_gray(frame)
            rects = engine.detect_faces(gray)
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            if rects and recording:
                saved += 1
                # The whole frame, not a crop: train.py re-runs the same
                # detect/crop pipeline that recognition runs on a live frame.
                cv2.imwrite(str(target / f"frame_{offset + saved:03d}.png"), gray)
            label = f"{saved}/{count}" + ("" if recording else "  [paused]")
            cv2.putText(frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return saved


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    key = slugify(args.name)
    target = Path(args.dataset) / key
    target.mkdir(parents=True, exist_ok=True)
    engine = FaceEngine()

    if args.images:
        saved = import_folder(engine, Path(args.images), target)
    else:
        saved = capture_from_camera(engine, target, args.camera, args.count)

    if not saved:
        print("[face] nothing was written; the dataset is unchanged.", file=sys.stderr)
        return 1

    print(f"[face] saved {saved} image(s) to {target}")
    if not args.no_register:
        store = PeopleStore()
        person = store.upsert(key, name=args.name, age=args.age, department=args.department)
        print(f"[face] registered '{person['name']}' as '{key}' in {store.path}")
    print("[face] next: python train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
