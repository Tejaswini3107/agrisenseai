import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

sys.modules['data_pipeline'] = MagicMock()
sys.modules['data_pipeline.climate_model'] = MagicMock()
sys.modules['data_pipeline.climate_model.predict'] = MagicMock()
sys.modules['data_pipeline.collectors'] = MagicMock()
sys.modules['data_pipeline.collectors.openweather'] = MagicMock()
sys.modules['data_pipeline.collectors.open_meteo'] = MagicMock()
sys.modules['data_pipeline.collectors.nasa_power'] = MagicMock()
sys.modules['ml'] = MagicMock()
sys.modules['ml.pest'] = MagicMock()
sys.modules['ml.pest.predict'] = MagicMock()
sys.modules['ml.irrigation'] = MagicMock()
sys.modules['ml.irrigation.predict'] = MagicMock()
sys.modules['bcrypt'] = MagicMock()

with patch('app.database.init_db', return_value=None), \
     patch('app.database.engine', MagicMock()), \
     patch('app.database.Base.metadata.create_all', return_value=None):

    from fastapi.testclient import TestClient  # noqa: E402
    from app.main import app  # noqa: E402
    from app.database import get_db  # noqa: E402

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


def test_crop_catalog():
    response = client.get("/api/crops")
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "enum"
    assert any(item["key"] == "rice" for item in payload["crops"])
    assert any(item["key"] == "sugarcane" for item in payload["crops"])
    assert all(item["key"] != "tomato" for item in payload["crops"])


@patch("app.routers.farmers.is_configured", return_value=True)
@patch("app.routers.farmers.verify_id_token", return_value={"uid": "google-uid-123", "email": "farmer@example.com", "name": "Farmer One", "picture": "https://example.com/avatar.png"})
def test_google_signin_creates_farmer(mock_verify_id_token, mock_is_configured):
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        response = client.post("/api/farmers/google-signin", json={"id_token": "token", "crop": "sugarcane"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "created"
        assert payload["farmer"]["email"] == "farmer@example.com"
        assert payload["farmer"]["current_crop"] == "sugarcane"
    finally:
        app.dependency_overrides.clear()


def test_google_auth_status_not_configured():
    response = client.get("/api/auth/google/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is False
    assert "GOOGLE_OAUTH_WEB_CLIENT_ID" in payload["client_id_source"]


def test_google_auth_status_from_env(monkeypatch):
    monkeypatch.setenv(
        "GOOGLE_OAUTH_WEB_CLIENT_ID",
        "dummy-google-web-client-id.apps.googleusercontent.com",
    )

    from app.services.auth_google import is_configured

    assert is_configured() is True


def test_notification_status():
    response = client.get("/api/notifications/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["roles"] == ["farmer", "admin"]


@patch("app.routers.notifications.is_push_configured", return_value=True)
@patch("app.routers.notifications.send_push_notification", return_value=MagicMock(responses=[MagicMock(success=True)]))
@patch("app.routers.notifications.send_email_message", return_value=None)
def test_emergency_alert_sends_push_and_email(mock_send_email_message, mock_send_push_notification, mock_is_configured):
    from app.database import NotificationDevice, RegisteredFarmer

    device = MagicMock(spec=NotificationDevice)
    device.token = "device-token-1"
    device.user_uid = "google-uid-123"
    device.preferred_language = "hindi"

    farmer = MagicMock(spec=RegisteredFarmer)
    farmer.email = "farmer@example.com"

    device_query = MagicMock()
    device_query.filter.return_value.all.return_value = [device]

    farmer_query = MagicMock()
    farmer_query.filter.return_value.first.return_value = farmer

    def query_side_effect(model):
        if model is NotificationDevice:
            return device_query
        if model is RegisteredFarmer:
            return farmer_query
        return MagicMock()

    mock_db = MagicMock()
    mock_db.query.side_effect = query_side_effect
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        response = client.post(
            "/api/notifications/emergency-alert",
            json={
                "roles": ["farmer"],
                "crop": "rice",
                "location": "Farm field",
                "detail": "Pest pressure is rising quickly",
                "severity": "critical",
                "language": "hindi",
                "send_email": True,
                "data": {"alert_type": "emergency"},
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "processed"
        assert payload["results"][0]["language"] == "hindi"
        assert payload["results"][0]["push"]["status"] == "sent"
        assert payload["results"][0]["email"]["status"] == "sent"
        mock_send_push_notification.assert_called_once()
        mock_send_email_message.assert_called_once()
    finally:
        app.dependency_overrides.clear()


@patch("app.routers.chatbot.get_current_weather", return_value={"temperature": 30, "humidity": 78, "rainfall_mm": 1.5, "condition": "Cloudy", "location_name": "Test Farm"})
@patch("app.routers.chatbot.get_forecast", return_value=[{"date": "2026-05-22", "temperature": 30, "humidity": 78, "rainfall_mm": 1.5, "condition": "Cloudy"}])
@patch("app.routers.chatbot.get_soil_moisture", return_value=42)
@patch("app.routers.chatbot.ask_agriparam", return_value="कृपया आज सिंचाई न करें।")
def test_irrigation_language(mock_ask_agriparam, mock_soil_moisture, mock_forecast, mock_current_weather):
    response = client.post("/api/chatbot/irrigation", params={"reason": "Rain expected", "language": "hindi"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["language"] == "hindi"
    assert payload["explanation"] == "कृपया आज सिंचाई न करें।"


@patch("app.main.get_current_weather", return_value={"temperature": 30, "humidity": 78, "rainfall_mm": 1.5, "condition": "Cloudy", "location_name": "Test Farm"})
@patch("app.main.get_forecast", return_value=[{"date": "2026-05-22", "temperature": 30, "humidity": 78, "rainfall_mm": 1.5, "condition": "Cloudy"}])
@patch("app.main.get_soil_moisture", return_value=42)
def test_pest_detection(mock_soil_moisture, mock_forecast, mock_current_weather):
    response = client.get("/api/pest-detection/paddy")
    assert response.status_code == 200
    payload = response.json()
    assert payload["crop"] == "rice"
    assert payload["crop_key"] == "rice"
    assert payload["severity"] in {"HIGH", "CRITICAL"}
    assert len(payload["recommendations"]) >= 3


@patch("app.routers.chatbot.get_current_weather", return_value={"temperature": 30, "humidity": 78, "rainfall_mm": 1.5, "condition": "Cloudy", "location_name": "Test Farm"})
@patch("app.routers.chatbot.get_forecast", return_value=[{"date": "2026-05-22", "temperature": 30, "humidity": 78, "rainfall_mm": 1.5, "condition": "Cloudy"}])
@patch("app.routers.chatbot.get_soil_moisture", return_value=42)
@patch("app.routers.chatbot.ask_agriparam", return_value="No irrigation needed because rain is expected.")
def test_irrigation_advice(mock_ask_agriparam, mock_soil_moisture, mock_forecast, mock_current_weather):
    response = client.post("/api/chatbot/irrigation", params={"irrigate": "false", "reason": "Rain expected"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision_hindi"] == "पानी मत दो"
    assert payload["decision"] == "wait before irrigating"
    assert len(payload["schedule"]) >= 1
