// Backend integration: call FastAPI endpoints exposed by agrisense-backend.
// The backend exposes:
//  GET /api/weather/current?lat=<>&lon=<>  -> returns current weather + model predictions
//  GET /api/weather/forecast?lat=<>&lon=<> -> returns forecast list
//  GET /api/weather/health                 -> returns service health
// The base URL is centralized in apiConfig so it is not duplicated across services.

import {API_BASE_URL} from './apiConfig';

const DEFAULT_LAT = 18.6650; // Bodhan, Nizamabad
const DEFAULT_LON = 77.9046;

const KC_TABLE = {
  rice: {seedling: 1.05, vegetative: 1.2, flowering: 1.2, maturity: 0.75},
  wheat: {seedling: 0.7, vegetative: 1.15, flowering: 1.15, maturity: 0.45},
  cotton: {seedling: 0.45, vegetative: 1.1, flowering: 1.2, maturity: 0.6},
  sugarcane: {seedling: 0.55, vegetative: 1.0, flowering: 1.25, maturity: 0.75},
};

const SUPPORTED_CROPS = ['rice', 'wheat', 'cotton', 'maize', 'turmeric', 'sugarcane', 'soybean', 'potato', 'onion', 'garlic'];

const mapBackendCurrentToHome = (currentData) => {
  if (!currentData) return {};

  return {
    temperature: currentData.temperature,
    humidity: currentData.humidity,
    rainfall_mm: currentData.rainfall_mm,
    wind_speed: currentData.wind_speed,
    condition: currentData.condition,
    location_name: currentData.location_name,
    soil_moisture: currentData.soil_moisture,
    recommended_crop: currentData.recommended_crop,
    crop_confidence: currentData.crop_confidence,
    disease_risk: currentData.disease_risk,
    disease_confidence: currentData.disease_confidence,
    plant_stress: currentData.plant_stress,
    stress_confidence: currentData.stress_confidence,
    irrigation_need_litres: currentData.irrigation_need_litres,
    expected_yield_tons: currentData.expected_yield_tons,
    climate_risk: currentData.climate_risk,
    climate_confidence: currentData.climate_confidence,
    today: {
      temperatureC: currentData.temperature != null ? parseFloat(currentData.temperature.toFixed(1)) : 0,
      humidityPct: Math.round(currentData.humidity ?? 0),
      rainfallMm: Math.round(currentData.rainfall_mm ?? 0),
      windKph: Math.round(currentData.wind_speed ?? 0),
      condition: currentData.condition || '—',
    },
  };
};

// Parse YYYY-MM-DD in local time — avoids UTC-midnight off-by-one (e.g. "Tue" showing as "Mon" in IST)
const parseLocalDate = (dateStr) => {
  if (!dateStr) return new Date();
  const [y, m, d] = dateStr.split('-').map(Number);
  return new Date(y, m - 1, d);
};

const mapBackendForecastToHome = (forecastList) => {
  if (!Array.isArray(forecastList)) return [];
  const todayStr = new Date().toISOString().slice(0, 10);
  return forecastList.map((item) => {
    const dateObj = parseLocalDate(item.date);
    const isToday = item.date === todayStr;
    return {
      day: isToday ? 'Today' : dateObj.toLocaleDateString('en-US', {weekday: 'short'}),
      date: item.date,
      temperatureC: item.temperature != null ? parseFloat(item.temperature.toFixed(1)) : 0,
      humidityPct: Math.round(item.humidity ?? 0),
      rainfallMm: Math.round(item.rainfall_mm ?? 0),
      windKph: Math.round(item.wind_speed ?? 0),
      condition: item.condition || 'partly cloudy',
    };
  });
};

const fetchJson = async (url) => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
};

const postJson = async (url, body) => {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  return response.json();
};

// ML endpoints (pest/predict, irrigation/predict) only accept these 4 crop types.
// Map all other crops to the closest supported one.
const ML_CROP_MAP = {
  rice: 'rice', wheat: 'wheat', cotton: 'cotton', sugarcane: 'sugarcane',
  maize: 'wheat', soybean: 'wheat', potato: 'wheat', onion: 'wheat',
  garlic: 'wheat', turmeric: 'rice',
};

const toSupportedCrop = (cropType) => {
  const normalized = (cropType || '').toLowerCase();
  return ML_CROP_MAP[normalized] ?? 'rice';
};

const toSupportedGrowthStage = (growthStage) => {
  const normalized = (growthStage || '').toLowerCase();
  if (['seedling', 'vegetative', 'flowering', 'maturity'].includes(normalized)) return normalized;
  return 'vegetative';
};

const toSoilMoisture = (soilFeel) => {
  const key = (soilFeel || '').toLowerCase();
  if (key === 'dry') return 20;
  if (key === 'wet') return 80;
  return 50;
};

// Lightweight ET0 estimate when dedicated ET0 feed is unavailable.
// For MVP this keeps irrigation endpoint fully usable from live weather data.
const estimateEt0 = ({temperature = 30, humidity = 60, wind_speed = 8}) => {
  const tempFactor = Math.max(0, temperature - 10) * 0.17;
  const humidityFactor = Math.max(0.5, 1 - humidity / 200);
  const windFactor = 1 + Math.min(wind_speed, 30) / 100;
  return Number((tempFactor * humidityFactor * windFactor).toFixed(2));
};

export const getKcForCropStage = ({cropType, growthStage}) => {
  const crop = toSupportedCrop(cropType);
  const stage = toSupportedGrowthStage(growthStage);
  return KC_TABLE[crop]?.[stage] ?? 1.0;
};

export const buildPestPayloadFromInputs = ({weather, cropType, growthStage, previousPestOccurrence}) => {
  return {
    temperature: Number(weather?.temperature ?? 0),
    humidity: Number(weather?.humidity ?? 0),
    rainfall_mm: Number(weather?.rainfall_mm ?? 0),
    wind_speed: Number(weather?.wind_speed ?? 0),
    crop_type: toSupportedCrop(cropType),
    growth_stage: toSupportedGrowthStage(growthStage),
    previous_pest_occurrence: previousPestOccurrence ? 1 : 0,
  };
};

export const buildIrrigationPayloadFromInputs = ({weather, cropType, growthStage, daysSinceIrrigation, soilFeel}) => {
  const safeGrowthStage = toSupportedGrowthStage(growthStage);
  const safeCropType = toSupportedCrop(cropType);
  const kc = getKcForCropStage({cropType: safeCropType, growthStage: safeGrowthStage});
  const et0 = estimateEt0(weather);

  return {
    temperature: Number(weather?.temperature ?? 0),
    humidity: Number(weather?.humidity ?? 0),
    rainfall_mm: Number(weather?.rainfall_mm ?? 0),
    soil_moisture: toSoilMoisture(soilFeel),
    days_since_irrigation: Number(daysSinceIrrigation ?? 0),
    et0,
    kc,
    crop_type: safeCropType,
    growth_stage: safeGrowthStage,
  };
};

export const getWeatherCurrent = async (lat = DEFAULT_LAT, lon = DEFAULT_LON) => {
  const currentUrl = `${API_BASE_URL}/api/weather/current?lat=${lat}&lon=${lon}`;
  return fetchJson(currentUrl);
};

export const getWeatherForecast = async (lat = DEFAULT_LAT, lon = DEFAULT_LON) => {
  const forecastUrl = `${API_BASE_URL}/api/weather/forecast?lat=${lat}&lon=${lon}`;
  return fetchJson(forecastUrl);
};

export const getWeatherHealth = async () => {
  const healthUrl = `${API_BASE_URL}/api/weather/health`;
  return fetchJson(healthUrl);
};

export const getCropCatalog = async () => {
  const cropsUrl = `${API_BASE_URL}/api/crops`;
  return fetchJson(cropsUrl);
};

export const googleSignInFarmer = async ({idToken, crop}) => {
  return postJson(`${API_BASE_URL}/api/farmers/google-signin`, {
    id_token: idToken,
    crop,
  });
};

export const recordFarmerCropSearch = async ({farmerId, crop, source = 'search'}) => {
  return postJson(`${API_BASE_URL}/api/farmers/${farmerId}/crop-search`, {
    crop,
    source,
  });
};

export const getPestDetection = async (crop = 'rice', lat = DEFAULT_LAT, lon = DEFAULT_LON) => {
  const params = new URLSearchParams({lat: String(lat), lon: String(lon)});
  return fetchJson(`${API_BASE_URL}/api/pest-detection/${crop}?${params.toString()}`);
};

export const predictPest = async (payload) => {
  return postJson(`${API_BASE_URL}/api/pest/predict`, payload);
};

// Full chatbot irrigation endpoint — returns next_watering_time, 3-day schedule, AI insights, conditions.
// Backend: POST /api/chatbot/irrigation (scalar query params on a POST endpoint)
export const getIrrigationChatbot = async ({crop = 'rice', language = 'english', lat = DEFAULT_LAT, lon = DEFAULT_LON} = {}) => {
  const params = new URLSearchParams({
    crop,
    language,
    lat: String(lat),
    lon: String(lon),
    irrigate: 'false',
    reason: 'live-field-check',
  });
  const response = await fetch(`${API_BASE_URL}/api/chatbot/irrigation?${params.toString()}`, {method: 'POST'});
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json();
};

// Simple ML-only irrigation prediction — returns irrigate yes/no + water amount
export const predictIrrigation = async (payload) => {
  return postJson(`${API_BASE_URL}/api/irrigation/predict`, payload);
};

export const askAgriparam = async ({question, context = null, language = 'english'}) => {
  return postJson(`${API_BASE_URL}/api/chatbot/ask`, {question, context, language});
};

// Weather farming advice — POST /api/chatbot/weather-advice (scalar query params)
export const getWeatherAdvice = async ({temperature, humidity, rainfallMm, language = 'english'}) => {
  const params = new URLSearchParams({
    temperature: String(temperature),
    humidity: String(humidity),
    rainfall_mm: String(rainfallMm),
    language,
  });
  const response = await fetch(`${API_BASE_URL}/api/chatbot/weather-advice?${params.toString()}`, {method: 'POST'});
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json();
};

export const getDashboardWeather = async (lat = DEFAULT_LAT, lon = DEFAULT_LON) => {
  const [currentData, forecastData] = await Promise.all([
    getWeatherCurrent(lat, lon),
    getWeatherForecast(lat, lon),
  ]);

  return {
    ...mapBackendCurrentToHome(currentData),
    forecast: mapBackendForecastToHome(forecastData?.forecast ?? []),
  };
};
