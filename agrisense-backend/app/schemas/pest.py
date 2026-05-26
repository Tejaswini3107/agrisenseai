from pydantic import BaseModel
from typing import Literal


class PestRequest(BaseModel):
    temperature: float
    humidity: float
    rainfall_mm: float
    wind_speed: float
    crop_type: Literal["rice", "wheat", "cotton", "sugarcane"]
    growth_stage: Literal["seedling", "vegetative", "flowering", "maturity"]
    previous_pest_occurrence: int


class PestResponse(BaseModel):
    risk_level: str
    pest_type: str
    confidence: float
