import io
import pytest
from fastapi.testclient import TestClient


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_detect_image_returns_verdict(client: TestClient, tiny_jpeg: bytes):
    resp = client.post(
        "/detect/image",
        files={"file": ("test.jpg", io.BytesIO(tiny_jpeg), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "verdict" in data
    assert data["verdict"] in ("real", "fake", "uncertain")
    assert 0.0 <= data["fake_probability"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0


def test_detect_image_wrong_type(client: TestClient, tiny_jpeg: bytes):
    resp = client.post(
        "/detect/image",
        files={"file": ("test.gif", io.BytesIO(tiny_jpeg), "image/gif")},
    )
    assert resp.status_code == 415


def test_detect_image_saved_in_history(client: TestClient, tiny_jpeg: bytes):
    resp = client.post(
        "/detect/image",
        files={"file": ("history_test.jpg", io.BytesIO(tiny_jpeg), "image/jpeg")},
    )
    assert resp.status_code == 200
    detection_id = resp.json()["id"]

    hist = client.get(f"/history/{detection_id}")
    assert hist.status_code == 200
    assert hist.json()["id"] == detection_id


def test_history_list(client: TestClient, tiny_jpeg: bytes):
    client.post(
        "/detect/image",
        files={"file": ("list_test.jpg", io.BytesIO(tiny_jpeg), "image/jpeg")},
    )
    resp = client.get("/history/?limit=5")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_stats_endpoint(client: TestClient):
    resp = client.get("/history/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_analyzed" in data
    assert "fake_rate" in data
    assert data["total_analyzed"] >= 0


def test_delete_detection(client: TestClient, tiny_jpeg: bytes):
    resp = client.post(
        "/detect/image",
        files={"file": ("delete_me.jpg", io.BytesIO(tiny_jpeg), "image/jpeg")},
    )
    det_id = resp.json()["id"]

    del_resp = client.delete(f"/history/{det_id}")
    assert del_resp.status_code == 204

    get_resp = client.get(f"/history/{det_id}")
    assert get_resp.status_code == 404
