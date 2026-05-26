# AgriSense AI Backend

FastAPI backend for AgriSense AI. This service powers weather aggregation, crop and crop-health endpoints, pest and irrigation explanations, Google auth, AgriParam chat, and database-backed farmer records.

## Overview

The backend combines:

- live weather lookups from Open-Meteo and OpenWeather
- NASA POWER soil moisture data
- AI and rule-based predictions for crops, disease risk, plant stress, irrigation, climate risk, and pest guidance
- Hugging Face AgriParam responses for English and Hindi plus other supported languages
- SQLite fallback for local development and PostgreSQL for production

## Entry Points

- [app/main.py](app/main.py) - main FastAPI application used by the repo
- [main.py](main.py) - alternate standalone entry point

## Main Features

- Current weather plus AI predictions
- 5-day weather forecast
- Readable forecast output and CSV export
- Crop catalog and crop-health endpoints
- Pest guidance and irrigation explanation endpoints
- AgriParam-backed chatbot answers
- Google auth status and token verification
- Farmer and notification database tables

## API Endpoints

### Health and docs

- `GET /health`
- `GET /docs`
- `GET /openapi.json`

### Weather

- `GET /api/weather/current?lat=LAT&lon=LON`
- `GET /api/weather/forecast?lat=LAT&lon=LON`
- `GET /api/weather/forecast-readable?lat=LAT&lon=LON&output=json|csv`

### Chatbot and AgriParam

- `POST /api/chatbot/ask`
- `POST /api/chatbot/irrigation`
- `POST /api/chatbot/weather-advice`

### Other application routes

- `GET /api/crops`
- `GET /api/crop-health`
- `GET /api/pest-detection/{crop}`
- `GET /api/auth/google/status`
- `POST /api/auth/google/verify`

## Weather Flow

The weather router uses this order:

1. Open-Meteo for current weather and forecast
2. OpenWeather fallback if Open-Meteo is unavailable
3. NASA POWER for soil moisture
4. `predict_all()` to generate the AI outputs

The current weather response includes:

- temperature
- humidity
- rainfall_mm
- wind_speed
- condition
- location_name
- soil_moisture
- recommended_crop
- crop_confidence
- disease_risk
- disease_confidence
- plant_stress
- stress_confidence
- irrigation_need_litres
- expected_yield_tons
- climate_risk
- climate_confidence

## Chatbot Flow

### `POST /api/chatbot/ask`

Accepts a question, optional context, and language. The backend forwards the prompt to AgriParam.

### `POST /api/chatbot/irrigation`

Builds a live context from weather, soil moisture, forecast rainfall, and crop type, then asks AgriParam to explain whether the farmer should irrigate or wait.

### `POST /api/chatbot/weather-advice`

Converts the supplied weather values into a short context and gets a farming recommendation from AgriParam.

## AgriParam Support

The backend uses Hugging Face inference with model `bharatgenai/AgriParam`.

Supported languages include:

- English
- Hindi
- Assamese
- Bengali
- Gujarati
- Kannada
- Malayalam
- Marathi
- Punjabi
- Tamil
- Telugu
- Urdu
- Arabic
- French

If `HF_TOKEN` is missing, the service will still boot but AgriParam responses may fail at runtime.

## Database

The backend uses PostgreSQL when all of these are set:

- `DB_HOST`
- `DB_USER`
- `DB_PASSWORD`

If those variables are missing, the app falls back to a local SQLite database file.

### Database tables

- `farmer_profiles`
- `notification_devices`
- `admin_users`
- `registered_farmers`
- `farmer_crop_searches`
- `agriparam_templates`

## Environment Variables

### Required for production or live integration

- `DB_HOST`
- `DB_USER`
- `DB_PASSWORD`
- `OPENWEATHER_API_KEY`
- `HF_TOKEN`
- `GOOGLE_OAUTH_WEB_CLIENT_ID`
- `GOOGLE_OAUTH_IOS_CLIENT_ID`
- `ONESIGNAL_APP_ID`
- `ONESIGNAL_API_KEY`

### Used by deployment

- `FIREBASE_SERVICE_ACCOUNT_JSON_B64`
- `EC2_HOST`
- `EC2_SSH_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Local Setup

```bash
cd agrisense-backend
pip install -r requirements.txt
python app/main.py
```

## Testing

```bash
cd agrisense-backend
pytest tests -v
```

Useful test files in this folder:

- `tests/test_main.py`
- `tests/test_pest_irrigation.py`
- `test_openweather.py`
- `test_nasa.py`
- `test_features.py`
- `test_cleaner.py`

## Data and ML Areas

- `data_pipeline/collectors/` - OpenWeather, Open-Meteo, NASA POWER collectors
- `data_pipeline/climate_model/` - crop, weather, and disease prediction pipeline
- `ml/` - irrigation and pest model logic
- `evaluation/validate_weather.py` - weather validation helper

## Production Notes

- The backend is designed to run behind Docker in production.
- The deployment workflow pushes images to ECR and starts the production compose stack on EC2.
- Keep credentials out of the repository and use environment variables or GitHub Secrets.

## Related Docs

- [Root README](../README.md)
- [.github/workflows/deploy.yml](../.github/workflows/deploy.yml)
