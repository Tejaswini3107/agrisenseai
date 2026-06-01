from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.agriparam import ask_agriparam
from app.services.context_builder import build_weather_context, build_irrigation_context

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
    answer = ask_agriparam(request.question, request.context)
    return {
        "question": request.question,
        "answer": answer,
        "language": request.language,
    }


@router.post("/irrigation")
def explain_irrigation(request: IrrigationRequest):
    if request.weather_context is not None:
        context = build_irrigation_context(request.irrigate, request.weather_context, request.reason)
    else:
        decision = "irrigate today" if request.irrigate else "wait before irrigating"
        context = f"Explain in simple terms why a farmer should {decision}. Reason: {request.reason}"

    answer = ask_agriparam("What is your advice based on this irrigation situation?", context)
    return {"irrigation_decision": request.irrigate, "reason": request.reason, "advice": answer}


@router.post("/weather-advice")
def weather_advice(request: WeatherAdviceRequest):
    context_str = build_weather_context(request.dict())
    question = "What farming activities should the farmer do today and what risks should they watch for?"
    answer = ask_agriparam(question, context_str)
    return {"context_used": context_str, "advice": answer, "language": "english"}
