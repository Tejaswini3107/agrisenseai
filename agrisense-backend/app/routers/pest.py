from fastapi import APIRouter, HTTPException
from app.schemas.pest import PestRequest, PestResponse
from ml.pest.predict import predict

router = APIRouter(prefix="/api/pest", tags=["pest"])


@router.post("/predict", response_model=PestResponse)
def predict_pest(data: PestRequest):
    try:
        result = predict(data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
