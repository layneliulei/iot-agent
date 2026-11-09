from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_rejects_empty():
    r = client.post("/chat", json={"q": ""})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail[0]["type"] == "string_too_short"
