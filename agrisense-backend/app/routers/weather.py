from fastapi import APIRouter, HTTPException
from data_pipeline.collectors.openweather import get_current_weather, get_forecast
from data_pipeline.collectors.nasa_power import get_soil_moisture
from data_pipeline.climate_model.predict import predict_all

# Create router with prefix and tags
router = APIRouter(prefix="/api/weather", tags=["Weather"])


@router.get("/current")
async def get_current_weather_with_predictions(lat: float, lon: float):
    """
    Get current weather, soil moisture, and AI predictions for a location.

    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate

    Returns:
        dict: Combined response with weather, soil moisture, and all predictions
            Fields: temperature, humidity, rainfall_mm, wind_speed, condition, location_name,
                   soil_moisture, recommended_crop, crop_confidence, disease_risk,
                   disease_confidence, plant_stress, stress_confidence,
                   irrigation_need_litres, expected_yield_tons, climate_risk, climate_confidence

    Raises:
        HTTPException: 503 Service Unavailable on any error
    """
    try:
        # Fetch current weather from OpenWeather
        weather_dict = get_current_weather(lat, lon)

        if not weather_dict or "error" in weather_dict:
            raise HTTPException(status_code=503, detail="Failed to fetch current weather data")

        # Fetch soil moisture from NASA POWER
        soil_moisture = get_soil_moisture(lat, lon)

        # Add soil moisture to weather dict for predictions
        weather_dict["soil_moisture"] = soil_moisture

        # Get all AI predictions
        predictions = predict_all(weather_dict)

        # Combine all data into single response
        response = {
            # Weather data
            "temperature": weather_dict.get("temperature"),
            "humidity": weather_dict.get("humidity"),
            "rainfall_mm": weather_dict.get("rainfall_mm"),
            "wind_speed": weather_dict.get("wind_speed"),
            "condition": weather_dict.get("condition"),
            "location_name": weather_dict.get("location_name"),
            "soil_moisture": soil_moisture,
            # Model predictions
            "recommended_crop": predictions.get("recommended_crop"),
            "crop_confidence": predictions.get("crop_confidence"),
            "disease_risk": predictions.get("disease_risk"),
            "disease_confidence": predictions.get("disease_confidence"),
            "plant_stress": predictions.get("plant_stress"),
            "stress_confidence": predictions.get("stress_confidence"),
            "irrigation_need_litres": predictions.get("irrigation_need_litres"),
            "expected_yield_tons": predictions.get("expected_yield_tons"),
            "climate_risk": predictions.get("climate_risk"),
            "climate_confidence": predictions.get("climate_confidence"),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /current endpoint: {e}")
        raise HTTPException(status_code=503, detail=f"Weather service error: {str(e)}")


@router.get("/forecast")
async def get_weather_forecast(lat: float, lon: float):
    """
    Get 5-day weather forecast for a location.

    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate

    Returns:
        list: List of 5 daily forecasts, each with:
            - date (YYYY-MM-DD)
            - temperature (°C)
            - humidity (%)
            - rainfall_mm (mm)
            - wind_speed (km/h)

    Raises:
        HTTPException: 503 Service Unavailable on any error
    """
    try:
        forecast_data = get_forecast(lat, lon)

        if not forecast_data:
            raise HTTPException(status_code=503, detail="Failed to fetch forecast data")

        return {"latitude": lat, "longitude": lon, "forecast": forecast_data}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /forecast endpoint: {e}")
        raise HTTPException(status_code=503, detail=f"Forecast service error: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for weather service.

    Returns:
        dict: Service status
    """
    try:
        return {"status": "ok", "service": "weather"}
    except Exception as e:
        print(f"Error in /health endpoint: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")
