"""
People metadata store
=====================

``people.json`` maps an internal *person key* - the folder name under
``backend/dataset/`` - to the details the React app renders::

    {
      "ramya_s": {
        "name": "Ramya S",
        "age": 21,
        "department": "Computer Technology"
      }
    }

Keys are compared case-insensitively with spaces, underscores and dashes
collapsed, so a dataset folder called ``Ramya S`` and a store entry
``ramya-s`` resolve to the same person.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

log = logging.getLogger("face.people")

_LOCK = threading.Lock()

_COMMENT = ("People metadata. Each key must match a folder name under "
            "backend/dataset/. Add entries with capture.py.")


def normalise_key(value: Any) -> str:
    """Lower-case and collapse spaces, underscores and dashes."""
    return re.sub(r"[\s_\-]+", " ", str(value).strip().lower())


class PeopleStore:
    """A small JSON-backed directory of the people the model knows about."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or config.PEOPLE_FILE)

    # ------------------------------------------------------------------ #
    # Raw access
    # ------------------------------------------------------------------ #
    def load(self) -> Dict[str, Dict[str, Any]]:
        """Every registered person, keyed by dataset folder name."""
        with _LOCK:
            if not self.path.exists():
                return {}
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{self.path} must contain a JSON object.")
        return {
            key: value
            for key, value in raw.items()
            if not str(key).startswith("_") and isinstance(value, dict)
        }

    def save(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Write the store back to ``people.json`` (keeping the help comment)."""
        with _LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {"_comment": _COMMENT}
            payload.update(data)
            self.path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #
    def get(self, key: Any) -> Optional[Dict[str, Any]]:
        """One person by key *or* by display name; ``None`` when unknown."""
        wanted = normalise_key(key)
        if not wanted:
            return None
        for stored_key, record in self.load().items():
            if normalise_key(stored_key) == wanted or normalise_key(record.get("name")) == wanted:
                return self._decorate(stored_key, record)
        return None

    def details_for(self, key: Any) -> Dict[str, Any]:
        """Always return a displayable person, registered or not."""
        person = self.get(key)
        if person is not None:
            return person
        return {"key": str(key), "name": str(key), "age": "—", "department": "—",
                "registered": False}

    def all(self) -> List[Dict[str, Any]]:
        """Everybody, sorted by key."""
        items = sorted(self.load().items(), key=lambda pair: pair[0].lower())
        return [self._decorate(key, record) for key, record in items]

    def upsert(self, key: str, **fields: Any) -> Dict[str, Any]:
        """Create or update one person and return the stored record."""
        data = self.load()
        record = dict(data.get(key, {}))
        record.update({name: value for name, value in fields.items() if value is not None})
        data[key] = record
        self.save(data)
        return self._decorate(key, record)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _decorate(key: str, record: Dict[str, Any]) -> Dict[str, Any]:
        person: Dict[str, Any] = {"key": key, "registered": True}
        person.update(record)
        person["name"] = str(person.get("name") or key)
        person.setdefault("age", "—")
        person.setdefault("department", "—")
        return person
