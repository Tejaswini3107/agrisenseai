from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.agriparam import ask_agriparam
from app.services.context_builder import build_weather_context, build_irrigation_context
from app.services.crops import normalize_crop
from data_pipeline.collectors.openweather import get_current_weather, get_forecast
from data_pipeline.collectors.nasa_power import get_soil_moisture

router = APIRouter(prefix="/api/chatbot", tags=["Chatbot"])


class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None
    language: Optional[str] = "english"


class WeatherAdviceRequest(BaseModel):
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall_mm: Optional[float] = None
    wind_speed: Optional[float] = None
    condition: Optional[str] = None
    location_name: Optional[str] = None
    soil_moisture: Optional[float] = None
    recommended_crop: Optional[str] = None
    disease_risk: Optional[str] = None
    plant_stress: Optional[str] = None
    irrigation_need_litres: Optional[float] = None
    expected_yield_tons: Optional[float] = None
    climate_risk: Optional[str] = None


class IrrigationRequest(BaseModel):
    irrigate: bool
    reason: str
    weather_context: Optional[dict] = None


@router.post("/ask")
def ask_question(request: ChatRequest):
    answer = ask_agriparam(request.question, request.context, request.language)
    return {
        "question": request.question,
        "answer": answer,
        "language": request.language,
    }


@router.post("/irrigation")
def explain_irrigation(
    irrigate: bool = False,
    reason: str = "",
    crop: str = "rice",
    language: str = "english",
    lat: float = 18.6725,
    lon: float = 78.0941,
):
    crop_key = normalize_crop(crop)
    weather_data = get_current_weather(lat, lon) or {}
    forecast_data = get_forecast(lat, lon) or []
    soil_moisture = get_soil_moisture(lat, lon)

    rainfall = float(weather_data.get("rainfall_mm") or 0)
    humidity = float(weather_data.get("humidity") or 0)
    temperature = float(weather_data.get("temperature") or 0)
    rain_next_24h = sum(float(item.get("rainfall_mm") or 0) for item in forecast_data[:2])

    soil_thresholds = {
        "rice": 35,
        "wheat": 28,
    }
    soil_threshold = soil_thresholds.get(crop_key, 30)
    should_irrigate = soil_moisture < soil_threshold and rainfall < 2 and rain_next_24h < 5
    decision = "irrigate today" if should_irrigate else "wait before irrigating"
    question = f"Explain in simple terms why a farmer growing {crop_key} should {decision}. Reason: {reason}"
    live_context = (
        f"Crop: {crop_key}. "
        f"Location: {weather_data.get('location_name') or 'current farm location'}. "
        f"Temperature {temperature}°C, humidity {humidity}%, rainfall {rainfall}mm, "
        f"soil moisture {soil_moisture}%, rain in next 24h {round(rain_next_24h, 1)}mm, "
        f"soil threshold for irrigation {soil_threshold}%."
    )
    answer = ask_agriparam(question, live_context, language)

    schedule = []
    for index, item in enumerate(forecast_data[:3]):
        day_label = "Today" if index == 0 else "Tomorrow" if index == 1 else item.get("date", "").split("-")[-1]
        rain_amount = float(item.get("rainfall_mm") or 0)
        should_water = rain_amount < 3 and should_irrigate
        schedule.append(
            {
                "label": day_label,
                "full_date": item.get("date") or day_label,
                "should_water": should_water,
                "status_label": "Water 6-8 AM" if should_water else "Don't Water",
            }
        )

    return {
        "decision": decision,
        "decision_hindi": "पानी दो" if should_irrigate else "पानी मत दो",
        "crop": crop_key,
        "language": language,
        "explanation": answer,
        "next_watering_time": "Tomorrow 6-8 AM" if should_irrigate else "After the next dry spell",
        "next_watering_hindi": "कल सुबह 6-8 बजे" if should_irrigate else "अगले सूखे समय के बाद",
        "conditions": [
            {"label": "Temperature", "value": f"{round(temperature)}°C", "sub": weather_data.get("condition") or "Live weather", "icon": "temperature"},
            {"label": "Humidity", "value": f"{round(humidity)}%", "sub": "Live reading", "icon": "humidity"},
            {"label": "Rain", "value": f"{round(rainfall)} mm", "sub": f"Next 24h {round(rain_next_24h, 1)} mm", "icon": "rain"},
            {"label": "Soil", "value": f"{round(soil_moisture, 1)}%", "sub": "NASA POWER", "icon": "wind"},
        ],
        "insights": [
            {
                "title": "Soil Moisture Analysis",
                "body": f"Soil moisture is {round(soil_moisture, 1)}%, so today's irrigation recommendation is based on live field conditions.",
            },
            {
                "title": "Rain Forecast",
                "body": f"Forecasted rainfall over the next 24 hours is {round(rain_next_24h, 1)} mm.",
            },
            {
                "title": "Water Saving",
                "body": "Live weather data is being used to reduce unnecessary watering and conserve water.",
            },
        ],
        "schedule": schedule,
        "live_weather": {
            "temperature": temperature,
            "humidity": humidity,
            "rainfall_mm": rainfall,
            "soil_moisture": soil_moisture,
            "location_name": weather_data.get("location_name"),
        },
    }


@router.post("/weather-advice")
def weather_advice(temperature: float, humidity: float, rainfall_mm: float, language: str = "english"):
    context = (
        f"Current weather: temperature {temperature}°C, "
        f"humidity {humidity}%, rainfall {rainfall_mm}mm"
    )
    question = "What farming activities should the farmer do today based on this weather?"
    answer = ask_agriparam(question, context, language)
    return {"weather_context": context, "advice": answer, "language": language}
