from fastapi import APIRouter, HTTPException
from app.schemas.irrigation import IrrigationRequest, IrrigationResponse
from ml.irrigation.predict import predict

router = APIRouter(prefix="/api/irrigation", tags=["irrigation"])


@router.post("/predict", response_model=IrrigationResponse)
def predict_irrigation(data: IrrigationRequest):
    try:
        result = predict(data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
