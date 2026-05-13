from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.agriparam import ask_agriparam

router = APIRouter(prefix="/api/chatbot", tags=["Chatbot"])


class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None
    language: Optional[str] = "english"


@router.post("/ask")
def ask_question(request: ChatRequest):
    answer = ask_agriparam(request.question, request.context)
    return {
        "question": request.question,
        "answer": answer,
        "language": request.language,
    }


@router.post("/irrigation")
def explain_irrigation(irrigate: bool, reason: str):
    decision = "irrigate today" if irrigate else "wait before irrigating"
    question = f"Explain in simple terms why a farmer should {decision}. Reason: {reason}"
    answer = ask_agriparam(question)
    return {"decision": decision, "explanation": answer}


@router.post("/weather-advice")
def weather_advice(temperature: float, humidity: float, rainfall_mm: float):
    context = (
        f"Current weather: temperature {temperature}°C, "
        f"humidity {humidity}%, rainfall {rainfall_mm}mm"
    )
    question = "What farming activities should the farmer do today based on this weather?"
    answer = ask_agriparam(question, context)
    return {"weather_context": context, "advice": answer}
