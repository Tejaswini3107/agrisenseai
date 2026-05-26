from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

from app.routers import weather  # noqa: E402
from app.routers import chatbot  # noqa: E402
from app.routers import farmers  # noqa: E402
from app.routers import pest  # noqa: E402
from app.routers import irrigation  # noqa: E402
from app.database import init_db  # noqa: E402
from app.routers import admin_auth  # noqa: E402
from app.routers import notifications  # noqa: E402
from app.services.auth_google import is_configured as is_google_auth_configured, verify_id_token  # noqa: E402
from app.services.crops import get_crop_catalog, normalize_crop, is_supported_crop  # noqa: E402
from data_pipeline.collectors.openweather import get_current_weather, get_forecast  # noqa: E402
from data_pipeline.collectors.nasa_power import get_soil_moisture  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(
    title="AgriSense AI Backend",
    description="Climate and agricultural prediction API for crop recommendations, disease risk, and yield forecasting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

app.include_router(weather.router)
app.include_router(chatbot.router)
app.include_router(farmers.router)
app.include_router(admin_auth.router)
app.include_router(pest.router)
app.include_router(irrigation.router)
app.include_router(notifications.router)


class GoogleTokenRequest(BaseModel):
    id_token: str


@app.get("/")
async def root():
    return {
        "name": "AgriSense AI Backend",
        "version": "1.0.0",
        "description": "Climate and agricultural prediction API",
        "endpoints": {
            "current_weather": "/api/weather/current?lat=<latitude>&lon=<longitude>",
            "forecast": "/api/weather/forecast?lat=<latitude>&lon=<longitude>",
            "crops": "/api/crops",
            "health": "/api/weather/health",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


@app.get("/health")
async def app_health():
    return {"status": "ok", "service": "agrisense-backend"}


@app.get("/api/auth/google/status")
async def google_auth_status():
    return {
        "configured": is_google_auth_configured(),
        "client_id_source": "GOOGLE_OAUTH_WEB_CLIENT_ID | GOOGLE_OAUTH_IOS_CLIENT_ID",
    }


@app.post("/api/auth/google/verify")
async def google_verify(token_request: GoogleTokenRequest):
    if not is_google_auth_configured():
        return {
            "status": "not_configured",
            "message": "Google OAuth client IDs are not configured in the backend yet.",
        }

    decoded_token = verify_id_token(token_request.id_token)
    return {
        "status": "verified",
        "uid": decoded_token.get("uid"),
        "email": decoded_token.get("email"),
        "name": decoded_token.get("name"),
        "claims": decoded_token,
    }


@app.get("/api/crops")
async def get_crops():
    return {"source": "enum", "crops": get_crop_catalog()}


@app.get("/api/pest-detection/{crop}")
async def get_pest_detection(crop: str, lat: float = 18.6725, lon: float = 78.0941):
    crop_key = normalize_crop(crop)
    try:
        weather_data = get_current_weather(lat, lon) or {}
        forecast_data = get_forecast(lat, lon) or []
        soil_moisture = get_soil_moisture(lat, lon)
    except Exception as exc:
        return {
            "crop": crop_key,
            "pest_name": "General Pest Pressure",
            "pests": ["General Pest Pressure"],
            "risk_level": f"Unavailable - {str(exc)}",
            "severity": "INFO",
            "expected_days": "Updating now",
            "location": "Live data unavailable",
            "crops": [crop.title()],
            "reported_ago": "Just now",
            "recommendation": "Retry after the weather service responds",
            "recommendations": ["Retry after the weather service responds"],
            "voice_hint": "Tap voice button to hear in Hindi",
        }

    temperature = float(weather_data.get("temperature") or 0)
    humidity = float(weather_data.get("humidity") or 0)
    rainfall = float(weather_data.get("rainfall_mm") or 0)
    rain_next_24h = sum(float(item.get("rainfall_mm") or 0) for item in forecast_data[:2])

    pest_map = {
        "rice": {
            "pest_name": "Brown Planthopper",
            "recommendations": [
                "Scout lower stems and leaf sheaths today",
                "Use chlorpyrifos only if threshold is crossed",
                "Keep water levels stable to reduce spread",
            ],
        },
        "wheat": {
            "pest_name": "Aphid",
            "recommendations": ["Inspect leaf undersides", "Use neem-based spray", "Monitor nitrogen levels"],
        },
        "cotton": {
            "pest_name": "Whitefly",
            "recommendations": ["Set yellow sticky traps", "Spray approved insecticide", "Avoid over-irrigation"],
        },
    }

    fallback = {
        "pest_name": "General Pest Pressure",
        "recommendations": ["Monitor crop regularly", "Keep field boundaries clean", "Check for early leaf damage"],
    }

    pest_data = pest_map.get(crop_key, fallback)
    supported_crop = crop_key if is_supported_crop(crop_key) else "rice"
    score = 35
    if humidity >= 70:
        score += 20
    if 24 <= temperature <= 34:
        score += 15
    if rainfall > 2 or rain_next_24h > 4:
        score += 10
    if soil_moisture and soil_moisture > 35:
        score += 5

    if score >= 75:
        severity = "CRITICAL"
    elif score >= 55:
        severity = "HIGH"
    elif score >= 40:
        severity = "WARNING"
    else:
        severity = "INFO"

    confidence = min(97, max(42, score + 10))
    expected_days = "1-3 days" if severity in {"CRITICAL", "HIGH"} else "4-7 days"
    location = weather_data.get("location_name") or f"Near {crop.title()} fields"
    live_recommendations = [
        f"Humidity {int(humidity)}% favors {pest_data['pest_name'].lower()} activity",
        f"Rain in next 24h: {round(rain_next_24h, 1)} mm",
        f"Soil moisture: {round(soil_moisture or 0, 1)}%",
    ]

    return {
        "crop": crop_key,
        "crop_key": supported_crop,
        "pest_name": pest_data["pest_name"],
        "pests": [pest_data["pest_name"]],
        "risk_level": f"{severity.title()} - {confidence}% confidence",
        "severity": severity,
        "expected_days": expected_days,
        "location": location,
        "crops": [supported_crop.title()],
        "reported_ago": "Updated just now",
        "recommendation": pest_data["recommendations"][0],
        "recommendations": pest_data["recommendations"] + live_recommendations,
        "live_weather": {
            "temperature": temperature,
            "humidity": humidity,
            "rainfall_mm": rainfall,
            "soil_moisture": soil_moisture,
        },
        "voice_hint": "Tap voice button to hear in Hindi",
    }


@app.get("/api/crop-health")
async def get_crop_health():
    return {
        "crops": [
            {"name": "rice", "health": 85, "status": "healthy"},
            {"name": "wheat", "health": 78, "status": "good"},
            {"name": "cotton", "health": 72, "status": "fair"},
            {"name": "sugarcane", "health": 90, "status": "excellent"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
