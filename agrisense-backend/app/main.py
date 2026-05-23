from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

from app.routers import weather  # noqa: E402
from app.routers import chatbot  # noqa: E402
from app.routers import farmers  # noqa: E402
from app.database import init_db  # noqa: E402
from app.routers import admin_auth  # noqa: E402

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


@app.get("/")
async def root():
    return {
        "name": "AgriSense AI Backend",
        "version": "1.0.0",
        "description": "Climate and agricultural prediction API",
        "endpoints": {
            "current_weather": "/api/weather/current?lat=<latitude>&lon=<longitude>",
            "forecast": "/api/weather/forecast?lat=<latitude>&lon=<longitude>",
            "health": "/api/weather/health",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


@app.get("/health")
async def app_health():
    return {"status": "ok", "service": "agrisense-backend"}


@app.get("/api/pest-detection/{crop}")
async def get_pest_detection(crop: str):
    return {
        "crop": crop,
        "pests": [],
        "risk_level": "low",
        "recommendation": "Monitor crop regularly"
    }


@app.get("/api/crop-health")
async def get_crop_health():
    return {
        "crops": [
            {"name": "rice", "health": 85, "status": "healthy"},
            {"name": "wheat", "health": 78, "status": "good"},
            {"name": "cotton", "health": 72, "status": "fair"},
            {"name": "tomato", "health": 90, "status": "excellent"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
