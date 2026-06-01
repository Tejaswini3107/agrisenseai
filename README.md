# AgriSense AI

An AI-powered agricultural advisory platform built for smallholder farmers in South Asia. Delivers real-time pest risk assessment, irrigation recommendations, crop recommendations, climate risk analysis, and an agricultural chatbot — through a mobile app backed by a REST API.

---

## What It Does

Farmers in regions like Punjab, Sindh and Maharashtra often lose a significant portion of their yield to pest outbreaks or incorrect watering — not because they don't work hard, but because they don't have access to agronomists. AgriSense AI puts that advice in their pocket.

A farmer opens the app, selects their crop and growth stage, and gets told:
- Whether pests are likely to hit and which pest to watch for
- Whether they need to irrigate today and how much water to apply
- What the climate risk looks like for their location
- Which crop is best suited to their current conditions

---

## Project Structure

```
agrisense-backend/       Python FastAPI backend + ML models
agrisense-mobile/        React Native mobile app
agrisense-admin-website/ Next.js admin dashboard
```

---

## Backend — AI Modules

### Pest Risk Prediction
- 11 years of NASA POWER weather data across 9 South Asian locations
- Pest-environment thresholds from peer-reviewed research (PMC11882091, PMC7564875, PMC4153587)
- Random Forest classifier — 86.6% risk accuracy, 91.9% pest type accuracy
- Endpoint: `POST /api/pest/predict`

### Irrigation Recommendation
- FAO-56 international crop water requirement standards (Allen et al. 1998)
- XGBoost classifier + regressor — 99.78% decision accuracy, water amount MAE 4.87mm
- Endpoint: `POST /api/irrigation/predict`

### Climate & Crop Models
- Crop recommendation, climate risk, disease risk, plant stress — Random Forest
- LSTM neural network for weather time-series forecasting
- Endpoint: `GET /api/weather/current?lat=&lon=`

### Chatbot
- Agricultural Q&A in plain language
- Endpoint: `POST /api/chatbot/ask`

Full backend documentation → [agrisense-backend/README.md](agrisense-backend/README.md)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, PostgreSQL |
| ML | scikit-learn, XGBoost, TensorFlow, pandas |
| Data | NASA POWER API, OpenWeatherMap, Open-Meteo |
| Mobile | React Native |
| Admin | Next.js |
| Infrastructure | Docker, GitHub Actions CI/CD |

---

## Getting Started

```bash
# Backend
cd agrisense-backend
source ../venv/bin/activate
pip install -r requirements.txt

# Train models (required — model files are not in git)
python ml/pest/train.py
python ml/irrigation/train.py

# Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

API docs available at `http://localhost:8001/docs`

---

## Team

| Member | Module |
|--------|--------|
| Rehan Shafique | Pest risk prediction, irrigation recommendation, API endpoints, test suite |
| Marryum | Climate risk model, crop recommendation, disease risk, LSTM weather forecasting |
| Tejaswini | Frontend integration |
| Jeet | Frontend integration |
