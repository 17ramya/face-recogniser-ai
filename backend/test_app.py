"""
Verification for the face-recognition backend
=============================================

::

    python -m pytest test_app.py -q

Every test runs on synthetic images generated with NumPy, so there is no
webcam, no dataset download and no network access involved. The checks cover
the metadata store, training/persistence of the LBPH model, the recognition
pipeline and the whole HTTP contract the React app depends on.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as app_module  # noqa: E402  (the Flask module)
import capture  # noqa: E402  (the enrolment CLI)
import config  # noqa: E402
from engine import FaceEngine, dataset_summary, iter_dataset  # noqa: E402
from people import PeopleStore, normalise_key  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def make_texture(kind: str, seed: int, size: int = 200) -> np.ndarray:
    """A deterministic greyscale image; the two kinds are far apart for LBPH."""
    rng = np.random.default_rng(seed)
    image = rng.integers(60, 170, size=(size, size), dtype=np.uint8)
    if kind == "vertical":
        image[:, ::20] = 240
        image[:, 1::20] = 240
    else:
        image[::20, :] = 240
        image[1::20, :] = 240
    return image


def write_dataset(root: Path, per_person: int = 6) -> Path:
    """``root/alice`` (vertical stripes) and ``root/bob`` (horizontal stripes)."""
    for index, (key, kind) in enumerate((("alice", "vertical"), ("bob", "horizontal"))):
        folder = root / key
        folder.mkdir(parents=True, exist_ok=True)
        for number in range(per_person):
            image = make_texture(kind, seed=index * 100 + number)
            assert cv2.imwrite(str(folder / f"img_{number}.jpg"), image)
    return root


def encode(image: np.ndarray) -> bytes:
    """PNG bytes of ``image``, ready to be uploaded."""
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def trained_engine(tmp_path: Path, dataset: Path) -> FaceEngine:
    engine = FaceEngine(
        model_path=tmp_path / "models" / "lbph_model.yml",
        labels_path=tmp_path / "models" / "labels.json",
    )
    engine.train(dataset)
    engine.save()
    return engine


# --------------------------------------------------------------------------- #
# Metadata store
# --------------------------------------------------------------------------- #
def test_normalise_key_collapses_separators() -> None:
    assert normalise_key("  Ramya S ") == "ramya s"
    assert normalise_key("ramya_s") == normalise_key("Ramya-S") == "ramya s"


def test_people_store_roundtrip(tmp_path: Path) -> None:
    store = PeopleStore(tmp_path / "people.json")
    assert store.load() == {}

    store.upsert("alice", name="Alice Rao", age=20, department="Computer Technology")
    person = store.get("Alice Rao")
    assert person is not None
    assert person["name"] == "Alice Rao" and person["age"] == 20
    assert person["registered"] is True

    store.upsert("alice", department="Information Technology")
    assert store.get("alice")["department"] == "Information Technology"
    assert store.get("alice")["age"] == 20, "unrelated fields must survive an update"

    assert store.all()[0]["key"] == "alice"
    assert PeopleStore(tmp_path / "people.json").get("ALICE") is not None


def test_details_for_unknown_person(tmp_path: Path) -> None:
    missing = PeopleStore(tmp_path / "people.json").details_for("ghost")
    assert missing["name"] == "ghost" and missing["registered"] is False


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def test_dataset_summary_and_iteration(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset", per_person=3)
    summary = dataset_summary(dataset)
    assert summary["people_count"] == 2
    assert summary["images"] == 6
    assert [person["images"] for person in summary["people"]] == [3, 3]
    assert [key for key, _ in iter_dataset(dataset)] == ["alice", "bob"]
    assert dataset_summary(tmp_path / "missing")["people_count"] == 0


def test_training_without_images_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "dataset"
    empty.mkdir()
    with pytest.raises(ValueError):
        FaceEngine().train(empty)


def test_engine_trains_recognises_and_reloads(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    engine = trained_engine(tmp_path, dataset)

    assert engine.is_ready is True
    assert engine.labels == ["alice", "bob"]

    alice = cv2.imread(str(dataset / "alice" / "img_0.jpg"), cv2.IMREAD_GRAYSCALE)
    bob = cv2.imread(str(dataset / "bob" / "img_3.jpg"), cv2.IMREAD_GRAYSCALE)

    alice_key, alice_distance = engine.classify_face(alice)
    bob_key, bob_distance = engine.classify_face(bob)
    assert alice_key == "alice" and bob_key == "bob"
    assert alice_distance < 1.0 and bob_distance < 1.0, "a training image must match itself"

    # A fresh engine must recover the same answers from disk.
    reloaded = FaceEngine(
        model_path=engine.model_path, labels_path=engine.labels_path, threshold=engine.threshold
    )
    assert reloaded.load() is True
    assert reloaded.labels == ["alice", "bob"]
    assert reloaded.classify_face(alice)[0] == "alice"
    assert reloaded.ensure_fresh() is False


def test_evaluate_reports_training_matches(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset")
    engine = trained_engine(tmp_path, dataset)
    evaluation = engine.evaluate(dataset)
    assert evaluation["images"] == 12
    assert evaluation["accuracy"] >= 0.5
    assert evaluation["mean_distance"] is not None


def test_predict_bytes_without_a_model(tmp_path: Path) -> None:
    engine = FaceEngine(
        model_path=tmp_path / "none" / "lbph_model.yml",
        labels_path=tmp_path / "none" / "labels.json",
    )
    result = engine.predict_bytes(encode(make_texture("vertical", 1)))
    assert result["recognized"] is False
    assert result["faces"] == 0
    assert set(result["person"]) >= {"name", "age", "department"}


def test_predict_bytes_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        FaceEngine().predict_bytes(b"this is not an image")


def test_predict_decodes_greyscale_exactly_like_training(tmp_path: Path) -> None:
    """Recognition must decode an upload the way ``train()`` read the dataset.

    ``train()`` uses ``cv2.imread(..., IMREAD_GRAYSCALE)``.  Decoding uploads to
    colour and converting back with ``BGR2GRAY`` is *not* the same decoder: on the
    real enrolled photo the two disagreed by up to 4 grey levels on 3% of the
    pixels, which moved the Haar box and rewrote the LBP codes - pushing that
    person's own face to distance 69.96 against a threshold of 70, one step away
    from being rejected as a stranger.
    """
    photo = tmp_path / "photo.jpg"
    assert cv2.imwrite(str(photo), make_texture("vertical", seed=7))

    on_disk = cv2.imread(str(photo), cv2.IMREAD_GRAYSCALE)
    uploaded = FaceEngine.decode_gray(photo.read_bytes())
    assert uploaded.ndim == 2, "uploads are decoded straight to greyscale"
    assert np.array_equal(uploaded, on_disk), "the two pixel paths must be identical"


def test_import_folder_copies_photos_untouched(tmp_path: Path) -> None:
    """Enrolment stores the picture as it is; ``train.py`` does the cropping.

    Cropping and equalising at enrolment made ``train()`` detect a face *inside*
    that crop and crop it a second time, so an enrolled person was compared
    against a different region of their own face (distance ~61-70).
    """
    source = tmp_path / "incoming"
    source.mkdir()
    picture = source / "portrait.png"
    assert cv2.imwrite(str(picture), make_texture("vertical", seed=5))

    target = tmp_path / "dataset" / "alice"
    target.mkdir(parents=True)
    assert capture.import_folder(FaceEngine(), source, target) == 1

    copied = target / "photo_001.png"
    assert copied.read_bytes() == picture.read_bytes(), "the photo must survive byte for byte"
    assert cv2.imread(str(copied)).shape == (200, 200, 3), "no cropping or resizing at enrolment"


def test_enrolled_photo_comes_back_as_the_same_person(tmp_path: Path) -> None:
    """The contract the whole demo rests on: enrol a photo, get that person back."""
    source = tmp_path / "incoming"
    source.mkdir()
    photo = source / "portrait.png"
    assert cv2.imwrite(str(photo), make_texture("vertical", seed=11))

    dataset = tmp_path / "dataset"
    (dataset / "alice").mkdir(parents=True)
    (dataset / "bob").mkdir(parents=True)
    assert capture.import_folder(FaceEngine(), source, dataset / "alice") == 1
    assert cv2.imwrite(str(dataset / "bob" / "photo_001.png"),
                       make_texture("horizontal", seed=12))

    engine = trained_engine(tmp_path, dataset)
    result = engine.predict_bytes(photo.read_bytes())
    assert result["recognized"] is True
    assert result["match_key"] == "alice"
    assert result["distance"] < config.MATCH_THRESHOLD / 2, (
        "an enrolled photo must not be a borderline call"
    )


# --------------------------------------------------------------------------- #
# HTTP contract - exactly what src/App.js depends on
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A Flask test client wired to a throw-away model, store and dataset."""
    dataset = write_dataset(tmp_path / "dataset")
    engine = trained_engine(tmp_path, dataset)

    store = PeopleStore(tmp_path / "people.json")
    store.upsert("alice", name="Alice Rao", age=20, department="Computer Technology")
    store.upsert("bob", name="Bob Kumar", age=22, department="Information Technology")
    engine.resolver = store.details_for

    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "FULL_IMAGE_FALLBACK", True)
    monkeypatch.setattr(app_module, "engine", engine)
    monkeypatch.setattr(app_module, "store", store)

    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as test_client:
        yield test_client


def upload(client: Any, image: np.ndarray) -> Any:
    """POST ``image`` to /api/storeimage the same way the browser does."""
    return client.post(
        "/api/storeimage",
        data={"image": (io.BytesIO(encode(image)), "capturedImage.jpg")},
        content_type="multipart/form-data",
    )


def test_root_lists_the_service(client: Any) -> None:
    body = client.get("/").get_json()
    assert body["service"] == "face-recogniser API"
    assert any("storeimage" in endpoint for endpoint in body["endpoints"])
    assert body["model_ready"] is True
    # The label list is shared with the landing page, so keep it in order.
    assert body["endpoints"][0] == "GET /api/health"
    assert body["app_url"] == "http://localhost:3000"


def test_root_serves_html_to_browsers_and_json_to_scripts(client: Any) -> None:
    """A browser on the API port must be told where the app is, not shown JSON."""
    browser = client.get(
        "/", headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    )
    assert browser.status_code == 200
    assert browser.mimetype == "text/html"
    page = browser.get_data(as_text=True)
    assert "not the app" in page
    assert "http://localhost:3000" in page
    assert "/api/health" in page
    assert page.startswith("<!DOCTYPE html>")

    # curl, axios and the test client default to */* and keep the JSON contract.
    scripted = client.get("/", headers={"Accept": "*/*"})
    assert scripted.mimetype == "application/json"
    assert scripted.get_json()["service"] == "face-recogniser API"
    assert client.get("/").mimetype == "application/json"


def test_health_reports_model_state(client: Any) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert body["labels"] == ["alice", "bob"]
    assert body["people_registered"] == 2
    assert body["threshold"] == pytest.approx(config.MATCH_THRESHOLD)
    assert "people_count" in body["dataset"]


def test_people_endpoint(client: Any) -> None:
    body = client.get("/api/people").get_json()
    assert body["count"] == 2
    assert body["people"][0]["name"] == "Alice Rao"


def test_dataset_endpoint(client: Any) -> None:
    body = client.get("/api/dataset").get_json()
    assert isinstance(body["directory"], str)
    assert "people_count" in body and "images" in body


def test_legacy_name_endpoint(client: Any) -> None:
    assert client.get("/name?name=alice").get_json()["name"] == "Alice Rao"
    assert client.get("/name").get_json()["name"] == "Alice Rao"
    assert client.get("/name?name=nobody").status_code == 404


def test_storeimage_recognises_an_enrolled_person(client: Any) -> None:
    response = upload(client, make_texture("vertical", 7))
    assert response.status_code == 200
    body = response.get_json()
    assert body["detection"] in {"haar", "full-image"}
    assert body["recognized"] is True
    assert body["match_key"] == "alice"
    assert body["person"]["name"] == "Alice Rao"
    assert body["person"]["age"] == 20
    assert body["person"]["department"] == "Computer Technology"
    assert Path(body["stored"]).exists(), "the last frame must be kept for debugging"


def test_storeimage_keeps_the_frame_in_the_format_it_arrived(client: Any) -> None:
    """The browser posts PNG because mirrored frames are flipped back losslessly.

    The debug copy must not lie about its format, and the previous frame must be
    replaced rather than left lying next to the new one.
    """
    jpeg = upload(client, make_texture("vertical", 7)).get_json()
    assert Path(jpeg["stored"]).name == "last_capture.jpg"

    ok, buffer = cv2.imencode(".png", make_texture("vertical", 7))
    assert ok
    body = client.post(
        "/api/storeimage",
        data={"image": (io.BytesIO(buffer.tobytes()), "capturedImage.png")},
        content_type="multipart/form-data",
    ).get_json()

    stored = Path(body["stored"])
    assert stored.name == "last_capture.png"
    assert stored.read_bytes() == buffer.tobytes(), "the frame is kept byte for byte"
    assert not (stored.parent / "last_capture.jpg").exists(), "only the newest frame is kept"


def test_storeimage_always_returns_a_renderable_person(client: Any) -> None:
    for image in (np.full((200, 200), 128, dtype=np.uint8), make_texture("horizontal", 99)):
        body = upload(client, image).get_json()
        person = body["person"]
        assert set(person) >= {"name", "age", "department"}, "App.js needs these three keys"
        assert isinstance(person["name"], str) and person["name"]
        assert isinstance(body["recognized"], bool)
        assert isinstance(body["message"], str) and body["message"]


def test_storeimage_rejects_bad_requests(client: Any) -> None:
    assert client.post("/api/storeimage", data={}).status_code == 400
    broken = client.post(
        "/api/storeimage",
        data={"image": (io.BytesIO(b"not an image at all"), "x.jpg")},
        content_type="multipart/form-data",
    )
    assert broken.status_code == 400 and "error" in broken.get_json()

    # The old src/sendrequest.js posted JSON instead of a picture.
    legacy = client.post("/api/storeimage", json={"name": "alice"}).get_json()
    assert legacy["person"]["name"] == "Alice Rao"


def test_reload_picks_up_the_model(client: Any) -> None:
    body = client.post("/api/reload").get_json()
    assert body["model_ready"] is True
    assert sorted(body["labels"]) == ["alice", "bob"]


def test_errors_are_json(client: Any) -> None:
    missing = client.get("/definitely-not-here")
    assert missing.status_code == 404
    assert missing.get_json()["error"] == "Not found"
    assert client.get("/api/storeimage").status_code == 405