from pydantic import BaseModel
from typing import Literal


class IrrigationRequest(BaseModel):
    temperature: float
    humidity: float
    rainfall_mm: float
    soil_moisture: float
    days_since_irrigation: int
    et0: float
    kc: float
    crop_type: Literal["rice", "wheat", "cotton", "sugarcane"]
    growth_stage: Literal["seedling", "vegetative", "flowering", "maturity"]


class IrrigationResponse(BaseModel):
    irrigate: str
    water_amount_mm: int
    etc_mm_per_day: float
