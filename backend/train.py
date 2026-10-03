"""
Train the face-recognition model
================================

Reads every photo under ``dataset/<person>/``, learns one LBPH histogram per
person and writes two artifacts::

    models/lbph_model.yml    the recognizer (OpenCV YAML)
    models/labels.json       label id -> person key

Usage::

    python train.py                          # dataset/ -> models/
    python train.py --evaluate               # ... and score the training photos
    python train.py --dataset data --models out --threshold 60
    python train.py --json                   # machine-readable report only

The running API picks a freshly written model up automatically (every request
checks the file timestamp), or immediately after ``POST /api/reload``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import config
from engine import FaceEngine, dataset_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train the LBPH face recognizer from dataset/<person>/*.jpg",
    )
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR,
                        help="folder holding one sub-folder per person (default: backend/dataset)")
    parser.add_argument("--models", type=Path, default=config.MODELS_DIR,
                        help="folder the model is written to (default: backend/models)")
    parser.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD,
                        help="LBPH distance accepted as a match (default: 70)")
    parser.add_argument("--evaluate", action="store_true",
                        help="also report how well the training photos match back")
    parser.add_argument("--json", action="store_true", help="print the report as JSON only")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    model_path = Path(args.models) / config.MODEL_PATH.name
    labels_path = Path(args.models) / config.LABELS_PATH.name
    engine = FaceEngine(model_path=model_path, labels_path=labels_path, threshold=args.threshold)

    if not args.json:
        summary = dataset_summary(args.dataset)
        print(f"[face] dataset : {summary['people_count']} people, "
              f"{summary['images']} images in {summary['directory']}")
        for person in summary["people"]:
            print(f"[face]   - {person['name']}: {person['images']} image(s)")

    try:
        report = engine.train(args.dataset)
    except ValueError as exc:
        print(f"[face] error: {exc}", file=sys.stderr)
        return 1

    saved = engine.save()
    report["saved"] = saved
    if args.evaluate:
        report["evaluation"] = engine.evaluate(args.dataset)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    detection = report["detection"]
    print(f"[face] trained  : {report['people']} people on {report['images']} images "
          f"(haar {detection['haar']}, whole-image {detection['full-image']})")
    for skip in report["skipped"]:
        print(f"[face] skipped  : {skip}")
    print(f"[face] wrote    : {saved['model']}")
    print(f"[face] wrote    : {saved['labels']}")

    if args.evaluate:
        evaluation = report["evaluation"]
        print(f"[face] self-test: {evaluation['accuracy']:.1%} of {evaluation['images']} "
              f"training images matched (mean distance {evaluation['mean_distance']})")
        for miss in evaluation["misses"]:
            print(f"[face]   miss    : {miss}")

    if report["people"] < 2:
        print("[face] warning : only one person is enrolled, so LBPH can only answer "
              "'this person' or 'unknown'.")

    print("[face] done    : the API reloads this model automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
