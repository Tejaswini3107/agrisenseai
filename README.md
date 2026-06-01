# AgriSense AI — Backend

AI-powered agricultural advisory platform for South Asian farmers. Provides real-time pest risk assessment, irrigation recommendations, crop recommendations, climate risk analysis, and an agricultural chatbot — all through a REST API.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [AI Modules](#ai-modules)
  - [Pest Risk Prediction](#pest-risk-prediction)
  - [Irrigation Recommendation](#irrigation-recommendation)
  - [Climate & Crop Models](#climate--crop-models)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Testing](#testing)
- [Model Performance](#model-performance)
- [Data Sources](#data-sources)
- [Team](#team)

---

## Overview

AgriSense AI is built for smallholder farmers in South Asia who need data-driven advice but don't have access to agronomists. The backend exposes machine learning models as REST API endpoints. A mobile or web frontend sends field conditions and receives predictions in plain language.

**Supported crops:** Rice, Wheat, Cotton, Sugarcane

**Supported regions:** India and Pakistan (9 agricultural locations covering major crop belts)

---

## Architecture

```
agrisense-backend/
├── app/                        FastAPI application
│   ├── main.py                 App entry point, router registration
│   ├── routers/                API route handlers
│   ├── schemas/                Pydantic request/response models
│   ├── services/               Chatbot and context builder
│   └── database.py             PostgreSQL connection and farmer profiles
│
├── ml/                         Rehan's AI module (pest + irrigation)
│   ├── pest/
│   │   ├── generate_dataset.py Dataset generation from NASA POWER API
│   │   ├── train.py            Random Forest model training
│   │   └── predict.py          Inference function called by the API
│   └── irrigation/
│       ├── generate_dataset.py Dataset generation with FAO-56 rules
│       ├── train.py            XGBoost model training
│       └── predict.py          Inference function called by the API
│
├── data_pipeline/              Marryum's climate + crop models
│   ├── climate_model/          Climate risk, crop recommendation, LSTM weather
│   └── collectors/             NASA POWER, OpenWeather, Open-Meteo API clients
│
├── models/                     Saved .pkl model files (not in git — run train scripts)
├── data/                       Generated CSV datasets
└── tests/                      Pytest test suite
```

---

## AI Modules

### Pest Risk Prediction

**Developed by:** Rehan Shafique

**Problem:** Farmers lose significant yield to pest outbreaks they cannot predict. Early warning allows preventive action.

**Approach:** Two Random Forest classifiers trained on 11 years of historical weather data from NASA POWER. Labels derived from peer-reviewed pest-environment thresholds.

#### Dataset Generation (`ml/pest/generate_dataset.py`)

- Fetches daily weather (temperature, humidity, rainfall, wind speed) from NASA POWER API
- 9 locations across India and Pakistan, 2015–2025 = **36,162 rows**
- Pest risk score computed from weighted weather conditions + growth stage vulnerability
- Pest type assigned deterministically using published temperature/humidity thresholds per crop
- Gaussian noise added to prevent deterministic overfitting

**Pest thresholds sourced from:**
| Pest | Source |
|------|--------|
| Stem borer (*Scirpophaga incertulas*) | Sanyal et al. 2025, PLoS One — PMC11882091 |
| Aphids (*Sitobion avenae*, *Aphis gossypii*) | Honěk & Martinková 2014, PLoS One — PMC4153587 |
| Whitefly (*Bemisia tabaci*) | Kanakala & Ghanim 2020, Insects — PMC7564875 |
| Bollworm (*Helicoverpa armigera*) | Virachack et al. 2018, J. Fac. Agric. Kyushu Univ. |
| Leaf miner (*Liriomyza* spp.) | Capinera, UF/IFAS EDIS EENY-255 |
| Shoot borer, Top borer, Pyrilla | Singh et al. Academia.edu/39823883; J. Asia-Pacific Entomol. 2021 |

#### Model Training (`ml/pest/train.py`)

- **Algorithm:** `RandomForestClassifier` (scikit-learn)
- **Why Random Forest:** Handles mixed numerical and categorical features without normalisation. Ensemble of 150 trees reduces overfitting. Provides feature importances for explainability.
- **Encoding:** `OrdinalEncoder` for crop type and growth stage — preserves natural stage ordering (seedling → vegetative → flowering → maturity)
- **Split:** 80/20 train/test, stratified on risk level
- **Models saved:** `pest_model.pkl`, `pest_type_model.pkl`, `pest_risk_encoder.pkl`, `pest_type_encoder.pkl`, `pest_feature_encoder.pkl`

**Feature importances (pest risk model):**
| Feature | Importance |
|---------|-----------|
| previous_pest_occurrence | 25.4% |
| temperature | 19.2% |
| humidity | 18.2% |
| growth_stage | 16.8% |
| wind_speed | 12.0% |
| rainfall_mm | 6.5% |
| crop_type | 1.9% |

---

### Irrigation Recommendation

**Developed by:** Rehan Shafique

**Problem:** Over-irrigation wastes water and damages soil. Under-irrigation reduces yield. Farmers need precise daily guidance.

**Approach:** XGBoost classifier for the yes/no irrigation decision + XGBoost regressor for the exact water amount. Rules based on FAO-56 international crop water requirement standards.

#### Dataset Generation (`ml/irrigation/generate_dataset.py`)

- Same NASA POWER weather data as pest module
- **FAO-56 Kc coefficients** (Allen et al. 1998) define crop water demand per stage
- **FAO-56 Table 22 depletion fractions** define soil moisture trigger thresholds
- **IRRI Knowledge Bank** water amounts per crop per irrigation event
- ET0 (reference evapotranspiration) estimated by season from FAO-56 climate zone tables
- ETc = ET0 × Kc — actual crop water demand
- ±10% Gaussian noise on water amounts to prevent data leakage
- **36,162 rows — 51.5% no / 48.5% yes** (natural split from applying rules to real weather)

**Irrigation thresholds by crop:**
| Crop | Moisture trigger | Rain skip (7-day) | Max days without water |
|------|-----------------|-------------------|------------------------|
| Rice | 80% (near saturation) | 30mm | 3–5 days |
| Wheat | 20–23% | 20mm | 11–22 days |
| Cotton | 18–23% | 25mm | 8–22 days |
| Sugarcane | 18–24% | 28mm | 8–17 days |

#### Model Training (`ml/irrigation/train.py`)

- **Algorithm:** `XGBClassifier` + `XGBRegressor` (XGBoost)
- **Why XGBoost over Random Forest:** Irrigation decisions follow precise numerical thresholds. XGBoost builds trees sequentially, each correcting previous errors — superior for learning exact numerical boundaries. Consistently outperforms Random Forest on structured tabular data.
- **Regressor trained only on irrigation-yes rows** — no-rows (amount=0) would skew regression predictions
- **Hyperparameters:** n_estimators=150, max_depth=6, learning_rate=0.1
- **Models saved:** `irrigation_model.pkl`, `irrigation_amount_model.pkl`, `irrigation_encoder.pkl`, `irrigation_feature_encoder.pkl`

---

### Climate & Crop Models

**Developed by:** Marryum (team member)

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Crop recommendation | Random Forest | Recommends best crop for given climate conditions |
| Climate risk | Random Forest | Assesses overall climate risk level |
| Disease risk | Random Forest | Predicts crop disease likelihood |
| Plant stress | Random Forest | Detects plant stress from weather signals |
| LSTM weather | LSTM (TensorFlow) | Time-series weather forecasting |

---

## API Endpoints

### Pest Prediction

```
POST /api/pest/predict
```

**Request body:**
```json
{
  "temperature": 32.0,
  "humidity": 80.0,
  "rainfall_mm": 5.0,
  "wind_speed": 12.0,
  "crop_type": "rice",
  "growth_stage": "vegetative",
  "previous_pest_occurrence": 1
}
```

**Response:**
```json
{
  "risk_level": "high",
  "pest_type": "stem_borer",
  "confidence": 0.9867
}
```

| Field | Values |
|-------|--------|
| `crop_type` | `rice` `wheat` `cotton` `sugarcane` |
| `growth_stage` | `seedling` `vegetative` `flowering` `maturity` |
| `previous_pest_occurrence` | `0` or `1` |
| `risk_level` (response) | `low` `medium` `high` |
| `confidence` (response) | 0.0 – 1.0 |

---

### Irrigation Recommendation

```
POST /api/irrigation/predict
```

**Request body:**
```json
{
  "temperature": 34.0,
  "humidity": 65.0,
  "rainfall_mm": 2.0,
  "soil_moisture": 30.0,
  "days_since_irrigation": 4,
  "et0": 6.5,
  "kc": 1.15,
  "crop_type": "rice",
  "growth_stage": "vegetative"
}
```

**Response:**
```json
{
  "irrigate": "yes",
  "water_amount_mm": 58,
  "etc_mm_per_day": 7.47
}
```

| Field | Description |
|-------|-------------|
| `et0` | Reference evapotranspiration mm/day — from weather API |
| `kc` | Crop coefficient — optional, defaults to 1.0 |
| `soil_moisture` | Current soil moisture % |
| `irrigate` (response) | `yes` or `no` |
| `water_amount_mm` (response) | mm of water to apply (0 if no) |
| `etc_mm_per_day` (response) | Crop actual water demand = ET0 × Kc |

---

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/weather/current?lat=&lon=` | Current weather + AI predictions for location |
| GET | `/api/weather/forecast?lat=&lon=` | Weather forecast |
| POST | `/api/chatbot/ask` | Agricultural Q&A chatbot |
| POST | `/api/farmers` | Register farmer profile |
| GET | `/api/farmers/{id}` | Get farmer profile |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive Swagger UI |

---

## Project Structure

```
ml/pest/
├── generate_dataset.py   Fetches NASA POWER data, scores pest risk, assigns pest type
├── train.py              Trains 2 Random Forest classifiers, saves 5 .pkl files
└── predict.py            Loads models at startup, exposes predict() for the API

ml/irrigation/
├── generate_dataset.py   Fetches NASA POWER data, applies FAO-56 rules, labels rows
├── train.py              Trains XGBoost classifier + regressor, saves 4 .pkl files
└── predict.py            Loads models at startup, calculates ETc, exposes predict()

app/routers/
├── pest.py               POST /api/pest/predict
├── irrigation.py         POST /api/irrigation/predict
├── weather.py            GET /api/weather/current and /forecast
├── chatbot.py            POST /api/chatbot/ask
├── farmers.py            CRUD farmer profiles
└── admin_auth.py         Admin authentication

app/schemas/
├── pest.py               PestRequest + PestResponse Pydantic models
└── irrigation.py         IrrigationRequest + IrrigationResponse Pydantic models

tests/
├── test_main.py          Health check and root endpoint tests
└── test_pest_irrigation.py  25 tests for pest and irrigation endpoints
```

---

## Setup & Installation

**Requirements:** Python 3.12, PostgreSQL (optional — app runs without it)

```bash
# Clone the repo
git clone https://github.com/AgriSenseAI/agrisenseai.git
cd agrisenseai/agrisense-backend

# Create and activate virtual environment
python3 -m venv ../venv
source ../venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

Model `.pkl` files are not stored in git. Run these once to generate them locally.

```bash
# Activate venv first
source ../venv/bin/activate

# 1. Generate pest dataset (fetches from NASA POWER API — takes a few minutes)
python ml/pest/generate_dataset.py

# 2. Train pest models
python ml/pest/train.py

# 3. Generate irrigation dataset
python ml/irrigation/generate_dataset.py

# 4. Train irrigation models
python ml/irrigation/train.py

# 5. Start the API server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Server will be live at `http://localhost:8001`
Interactive docs at `http://localhost:8001/docs`

---

## Testing

```bash
# Run all pest and irrigation tests
python -m pytest tests/test_pest_irrigation.py -v

# Run only pest tests
python -m pytest tests/test_pest_irrigation.py -v -k "pest"

# Run only irrigation tests
python -m pytest tests/test_pest_irrigation.py -v -k "irrigation"

# Run all tests
python -m pytest tests/ -v
```

**Test coverage (25 tests):**
- Valid inputs return 200 and correct response fields
- Risk level is always one of: low, medium, high
- Confidence is always between 0.0 and 1.0
- Water amount is 0 when irrigate is no
- Water amount is positive when irrigate is yes
- ETc = ET0 × Kc verified mathematically
- All 4 crops tested individually
- All 4 growth stages tested individually
- Invalid crop type returns 422
- Invalid growth stage returns 422
- Missing required field returns 422
- Extreme conditions (dry soil, wet soil, high temperature) handled correctly

---

## Model Performance

| Model | Algorithm | Metric | Result |
|-------|-----------|--------|--------|
| Pest Risk | Random Forest | Accuracy | 86.59% |
| Pest Type | Random Forest | Accuracy | 91.87% |
| Irrigation Decision | XGBoost Classifier | Accuracy | 99.78% |
| Water Amount | XGBoost Regressor | MAE | 4.87 mm |
| Water Amount | XGBoost Regressor | R² | 0.75 |

**Note on irrigation accuracy:** 99.78% is genuine — the irrigation decision follows clear numerical thresholds (soil moisture, days since irrigation, rainfall) which XGBoost learns very precisely. It is not data leakage — confirmed by per-class classification report showing both yes and no predicted correctly.

**Note on R² = 0.75:** ±10% Gaussian noise was intentionally added to water amounts during dataset generation to prevent R²=1.0 (data leakage). R²=0.75 is the honest result reflecting real-world variability in water application.

---

## Data Sources

| Source | Used for |
|--------|---------|
| NASA POWER API (`power.larc.nasa.gov`) | 11 years of daily weather data (temperature, humidity, rainfall, wind speed) for 9 South Asian locations |
| FAO Irrigation and Drainage Paper No. 56 (Allen et al. 1998) | Kc crop coefficients, soil moisture depletion fractions, ET0 calculation |
| IRRI Knowledge Bank | Water application amounts per crop and growth stage |
| OpenWeatherMap API | Real-time weather for live predictions |
| Open-Meteo API | Weather forecasting |

---

## Team

| Member | Module |
|--------|--------|
| Rehan Shafique | Pest risk prediction, irrigation recommendation, API endpoints, test suite |
| Marryum | Climate risk model, crop recommendation model, disease risk model, LSTM weather forecasting, weather data pipeline |
