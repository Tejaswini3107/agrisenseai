// Backend integration: call FastAPI endpoints exposed by agrisense-backend.
// The backend exposes:
//  GET /api/weather/current?lat=<>&lon=<>  -> returns current weather + model predictions
//  GET /api/weather/forecast?lat=<>&lon=<> -> returns forecast list
//  GET /api/weather/health                 -> returns service health
// Configure the host per platform so the app talks to the real backend.

import {Platform} from 'react-native';

const BACKEND_BASE = Platform.select({
  android: 'http://10.0.2.2:8000',
  ios: 'http://localhost:9000',
  default: 'http://localhost:9000',
});

const DEFAULT_LAT = 18.6725;
const DEFAULT_LON = 78.0941;

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
      temperatureC: Math.round(currentData.temperature ?? 0),
      humidityPct: Math.round(currentData.humidity ?? 0),
      rainfallMm: Math.round(currentData.rainfall_mm ?? 0),
      windKph: Math.round(currentData.wind_speed ?? 0),
      condition: currentData.condition || '—',
    },
  };
};

const mapBackendForecastToHome = (forecastList) => {
  if (!Array.isArray(forecastList)) return [];
  return forecastList.map((item) => ({
    day: new Date(item.date).toLocaleDateString('en-US', { weekday: 'short' }),
    date: item.date,
    temperatureC: Math.round(item.temperature ?? 0),
    humidityPct: Math.round(item.humidity ?? 0),
    rainfallMm: Math.round(item.rainfall_mm ?? 0),
    windKph: Math.round(item.wind_speed ?? 0),
    condition: item.condition || 'Clear',
  }));
};

const fetchJson = async (url) => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
};

export const getWeatherCurrent = async (lat = DEFAULT_LAT, lon = DEFAULT_LON) => {
  const currentUrl = `${BACKEND_BASE}/api/weather/current?lat=${lat}&lon=${lon}`;
  return fetchJson(currentUrl);
};

export const getWeatherForecast = async (lat = DEFAULT_LAT, lon = DEFAULT_LON) => {
  const forecastUrl = `${BACKEND_BASE}/api/weather/forecast?lat=${lat}&lon=${lon}`;
  return fetchJson(forecastUrl);
};

export const getWeatherHealth = async () => {
  const healthUrl = `${BACKEND_BASE}/api/weather/health`;
  return fetchJson(healthUrl);
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
