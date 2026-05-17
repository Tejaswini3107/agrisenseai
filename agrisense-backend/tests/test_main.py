import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
