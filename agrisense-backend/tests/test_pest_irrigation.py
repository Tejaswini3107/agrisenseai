import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

sys.modules['data_pipeline'] = MagicMock()
sys.modules['data_pipeline.climate_model'] = MagicMock()
sys.modules['data_pipeline.climate_model.predict'] = MagicMock()
sys.modules['data_pipeline.collectors'] = MagicMock()
sys.modules['data_pipeline.collectors.openweather'] = MagicMock()
sys.modules['data_pipeline.collectors.nasa_power'] = MagicMock()
sys.modules['data_pipeline.collectors.open_meteo'] = MagicMock()

with patch('app.database.init_db', return_value=None), \
     patch('app.database.engine', MagicMock()), \
     patch('app.database.Base.metadata.create_all', return_value=None):

    from fastapi.testclient import TestClient
    from app.main import app

client = TestClient(app)

VALID_PEST = {
    "temperature": 32.0,
    "humidity": 80.0,
    "rainfall_mm": 5.0,
    "wind_speed": 12.0,
    "crop_type": "rice",
    "growth_stage": "vegetative",
    "previous_pest_occurrence": 1,
}

VALID_IRRIGATION = {
    "temperature": 34.0,
    "humidity": 65.0,
    "rainfall_mm": 2.0,
    "soil_moisture": 30.0,
    "days_since_irrigation": 4,
    "et0": 6.5,
    "kc": 1.15,
    "crop_type": "rice",
    "growth_stage": "vegetative",
}


# --- Pest tests ---

def test_pest_returns_200():
    r = client.post("/api/pest/predict", json=VALID_PEST)
    assert r.status_code == 200


def test_pest_response_fields():
    r = client.post("/api/pest/predict", json=VALID_PEST)
    body = r.json()
    assert "risk_level" in body
    assert "pest_type" in body
    assert "confidence" in body


def test_pest_risk_level_valid():
    r = client.post("/api/pest/predict", json=VALID_PEST)
    assert r.json()["risk_level"] in ("low", "medium", "high")


def test_pest_confidence_range():
    r = client.post("/api/pest/predict", json=VALID_PEST)
    conf = r.json()["confidence"]
    assert 0.0 <= conf <= 1.0


def test_pest_low_risk_conditions():
    payload = {**VALID_PEST, "temperature": 18.0, "humidity": 40.0,
               "rainfall_mm": 0.0, "previous_pest_occurrence": 0}
    r = client.post("/api/pest/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["risk_level"] in ("low", "medium", "high")


def test_pest_all_crops():
    for crop in ("rice", "wheat", "cotton", "sugarcane"):
        payload = {**VALID_PEST, "crop_type": crop}
        r = client.post("/api/pest/predict", json=payload)
        assert r.status_code == 200, f"Failed for crop: {crop}"


def test_pest_all_growth_stages():
    for stage in ("seedling", "vegetative", "flowering", "maturity"):
        payload = {**VALID_PEST, "growth_stage": stage}
        r = client.post("/api/pest/predict", json=payload)
        assert r.status_code == 200, f"Failed for stage: {stage}"


def test_pest_invalid_crop_type():
    payload = {**VALID_PEST, "crop_type": "banana"}
    r = client.post("/api/pest/predict", json=payload)
    assert r.status_code == 422


def test_pest_invalid_growth_stage():
    payload = {**VALID_PEST, "growth_stage": "harvested"}
    r = client.post("/api/pest/predict", json=payload)
    assert r.status_code == 422


def test_pest_missing_field():
    payload = {k: v for k, v in VALID_PEST.items() if k != "temperature"}
    r = client.post("/api/pest/predict", json=payload)
    assert r.status_code == 422


def test_pest_extreme_temperature():
    payload = {**VALID_PEST, "temperature": 50.0, "humidity": 95.0}
    r = client.post("/api/pest/predict", json=payload)
    assert r.status_code == 200


def test_pest_no_previous_occurrence():
    payload = {**VALID_PEST, "previous_pest_occurrence": 0}
    r = client.post("/api/pest/predict", json=payload)
    assert r.status_code == 200


# --- Irrigation tests ---

def test_irrigation_returns_200():
    r = client.post("/api/irrigation/predict", json=VALID_IRRIGATION)
    assert r.status_code == 200


def test_irrigation_response_fields():
    r = client.post("/api/irrigation/predict", json=VALID_IRRIGATION)
    body = r.json()
    assert "irrigate" in body
    assert "water_amount_mm" in body
    assert "etc_mm_per_day" in body


def test_irrigation_decision_valid():
    r = client.post("/api/irrigation/predict", json=VALID_IRRIGATION)
    assert r.json()["irrigate"] in ("yes", "no")


def test_irrigation_water_zero_when_no():
    payload = {
        **VALID_IRRIGATION,
        "soil_moisture": 80.0,
        "days_since_irrigation": 0,
        "rainfall_mm": 20.0,
    }
    r = client.post("/api/irrigation/predict", json=payload)
    body = r.json()
    assert r.status_code == 200
    if body["irrigate"] == "no":
        assert body["water_amount_mm"] == 0


def test_irrigation_water_positive_when_yes():
    r = client.post("/api/irrigation/predict", json=VALID_IRRIGATION)
    body = r.json()
    if body["irrigate"] == "yes":
        assert body["water_amount_mm"] > 0


def test_irrigation_etc_calculated():
    r = client.post("/api/irrigation/predict", json=VALID_IRRIGATION)
    body = r.json()
    expected_etc = round(VALID_IRRIGATION["et0"] * VALID_IRRIGATION["kc"], 2)
    assert abs(body["etc_mm_per_day"] - expected_etc) < 0.01


def test_irrigation_all_crops():
    for crop in ("rice", "wheat", "cotton", "sugarcane"):
        payload = {**VALID_IRRIGATION, "crop_type": crop}
        r = client.post("/api/irrigation/predict", json=payload)
        assert r.status_code == 200, f"Failed for crop: {crop}"


def test_irrigation_all_growth_stages():
    for stage in ("seedling", "vegetative", "flowering", "maturity"):
        payload = {**VALID_IRRIGATION, "growth_stage": stage}
        r = client.post("/api/irrigation/predict", json=payload)
        assert r.status_code == 200, f"Failed for stage: {stage}"


def test_irrigation_invalid_crop():
    payload = {**VALID_IRRIGATION, "crop_type": "mango"}
    r = client.post("/api/irrigation/predict", json=payload)
    assert r.status_code == 422


def test_irrigation_invalid_growth_stage():
    payload = {**VALID_IRRIGATION, "growth_stage": "dormant"}
    r = client.post("/api/irrigation/predict", json=payload)
    assert r.status_code == 422


def test_irrigation_missing_field():
    payload = {k: v for k, v in VALID_IRRIGATION.items() if k != "soil_moisture"}
    r = client.post("/api/irrigation/predict", json=payload)
    assert r.status_code == 422


def test_irrigation_dry_conditions():
    payload = {
        **VALID_IRRIGATION,
        "soil_moisture": 10.0,
        "days_since_irrigation": 10,
        "rainfall_mm": 0.0,
    }
    r = client.post("/api/irrigation/predict", json=payload)
    assert r.status_code == 200


def test_irrigation_wet_conditions():
    payload = {
        **VALID_IRRIGATION,
        "soil_moisture": 90.0,
        "days_since_irrigation": 1,
        "rainfall_mm": 30.0,
    }
    r = client.post("/api/irrigation/predict", json=payload)
    assert r.status_code == 200
