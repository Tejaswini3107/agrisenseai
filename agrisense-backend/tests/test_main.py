import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

sys.modules['data_pipeline'] = MagicMock()
sys.modules['data_pipeline.climate_model'] = MagicMock()
sys.modules['data_pipeline.climate_model.predict'] = MagicMock()
sys.modules['data_pipeline.collectors'] = MagicMock()
sys.modules['data_pipeline.collectors.openweather'] = MagicMock()
sys.modules['data_pipeline.collectors.nasa_power'] = MagicMock()

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "AgriSense AI Backend"


def test_crop_health():
    response = client.get("/api/crop-health")
    assert response.status_code == 200
    assert "crops" in response.json()


def test_pest_detection():
    response = client.get("/api/pest-detection/rice")
    assert response.status_code == 200
    assert response.json()["crop"] == "rice"
