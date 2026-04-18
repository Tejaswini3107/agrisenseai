import joblib
import numpy as np
import math
from tensorflow.keras.models import load_model

# Load all models and encoders at module level
print("Loading prediction models and encoders...")

try:
    # Crop recommendation model
    crop_model = joblib.load('models/crop_selector_model.pkl')
    crop_encoder = joblib.load('models/crop_encoder.pkl')
    
    # Disease risk model
    disease_model = joblib.load('models/disease_risk_model.pkl')
    disease_encoder = joblib.load('models/disease_encoder.pkl')
    
    # Plant stress model
    stress_model = joblib.load('models/plant_stress_model.pkl')
    stress_encoder = joblib.load('models/stress_encoder.pkl')
    
    # Irrigation requirement model
    irrigation_model = joblib.load('models/irrigation_requirement_model.pkl')
    
    # Yield prediction models
    yield_model = joblib.load('models/climate_model.pkl')  # Crop yield
    
    # Climate risk model
    climate_model = joblib.load('models/climate_risk_model.pkl')
    climate_encoder = joblib.load('models/climate_risk_encoder.pkl')
    
    # Feature scaler
    scaler = joblib.load('models/feature_scaler.pkl')
    
    # LSTM weather model
    lstm_model = load_model('models/lstm_weather_model.keras')
    
    print("✓ All models and encoders loaded successfully")
    
except FileNotFoundError as e:
    print(f"✗ Error loading model files: {e}")
    print("Please ensure all model files are in the models/ directory")
    raise
except Exception as e:
    print(f"✗ Error loading models: {e}")
    raise


def build_feature_vector(weather_dict):
    """
    Build feature vector from weather data.
    
    Args:
        weather_dict (dict): Dictionary with keys: temperature, humidity, rainfall_mm, wind_speed
    
    Returns:
        np.ndarray: Scaled feature vector of shape (1, 9)
    """
    T = weather_dict.get('temperature', 25.0)
    H = weather_dict.get('humidity', 65.0)
    R = weather_dict.get('rainfall_mm', 0.0)
    W = weather_dict.get('wind_speed', 10.0)
    
    # Compute heat index (°C)
    # Formula: -8.78469 + 1.61139*T + 2.33855*H - 0.14611*T*H - 0.01230*T^2 - 0.01642*H^2
    heat_index = -8.78469 + 1.61139*T + 2.33855*H - 0.14611*T*H - 0.01230*(T**2) - 0.01642*(H**2)
    heat_index = np.clip(heat_index, 10, 80)  # Clip to reasonable range
    
    # Compute dew point (°C)
    dew_point = T - (100 - H) / 5.0
    
    # Compute vapor pressure deficit (kPa)
    # VPD = 0.6108 * exp(17.27*T/(T+237.3)) * (1 - H/100)
    vpd = 0.6108 * math.exp(17.27*T/(T+237.3)) * (1 - H/100)
    
    # Binary features
    is_high_humidity = 1 if H > 70 else 0
    is_high_temp = 1 if T > 32 else 0
    
    # Build feature vector in exact order
    features = np.array([
        T,                      # temperature
        H,                      # humidity
        R,                      # rainfall_mm
        W,                      # wind_speed
        heat_index,             # heat_index
        dew_point,              # dew_point
        vpd,                    # vapor_pressure_deficit
        is_high_humidity,       # is_high_humidity
        is_high_temp            # is_high_temp
    ]).reshape(1, -1)
    
    # Apply scaler
    scaled_features = scaler.transform(features)
    
    return scaled_features


def predict_all(weather_dict):
    """
    Run all prediction models and return comprehensive results.
    
    Args:
        weather_dict (dict): Dictionary with keys: temperature, humidity, rainfall_mm, wind_speed
    
    Returns:
        dict: Predictions from all models with keys:
            - recommended_crop (string)
            - crop_confidence (float)
            - disease_risk (string)
            - disease_confidence (float)
            - plant_stress (string)
            - stress_confidence (float)
            - irrigation_need_litres (float)
            - expected_yield_tons (float)
            - climate_risk (string)
            - climate_confidence (float)
    """
    # Build feature vector
    X = build_feature_vector(weather_dict)
    
    predictions = {}
    
    # 1. Crop recommendation
    try:
        crop_pred_proba = crop_model.predict_proba(X)[0]
        crop_pred_class = crop_model.predict(X)[0]
        crop_confidence = float(np.max(crop_pred_proba))
        recommended_crop = crop_encoder.inverse_transform([crop_pred_class])[0]
        
        predictions['recommended_crop'] = recommended_crop
        predictions['crop_confidence'] = round(crop_confidence, 4)
    except Exception as e:
        print(f"Error predicting crop: {e}")
        predictions['recommended_crop'] = 'Wheat'
        predictions['crop_confidence'] = 0.0
    
    # 2. Disease risk
    try:
        disease_pred_proba = disease_model.predict_proba(X)[0]
        disease_pred_class = disease_model.predict(X)[0]
        disease_confidence = float(np.max(disease_pred_proba))
        disease_risk = disease_encoder.inverse_transform([disease_pred_class])[0]
        
        predictions['disease_risk'] = disease_risk
        predictions['disease_confidence'] = round(disease_confidence, 4)
    except Exception as e:
        print(f"Error predicting disease risk: {e}")
        predictions['disease_risk'] = 'Low'
        predictions['disease_confidence'] = 0.0
    
    # 3. Plant stress
    try:
        stress_pred_proba = stress_model.predict_proba(X)[0]
        stress_pred_class = stress_model.predict(X)[0]
        stress_confidence = float(np.max(stress_pred_proba))
        plant_stress = stress_encoder.inverse_transform([stress_pred_class])[0]
        
        predictions['plant_stress'] = plant_stress
        predictions['stress_confidence'] = round(stress_confidence, 4)
    except Exception as e:
        print(f"Error predicting plant stress: {e}")
        predictions['plant_stress'] = 'Low'
        predictions['stress_confidence'] = 0.0
    
    # 4. Irrigation requirement
    try:
        irrigation_need_litres = float(irrigation_model.predict(X)[0])
        predictions['irrigation_need_litres'] = round(irrigation_need_litres, 1)
    except Exception as e:
        print(f"Error predicting irrigation: {e}")
        predictions['irrigation_need_litres'] = 0.0
    
    # 5. Expected yield
    try:
        expected_yield_tons = float(yield_model.predict(X)[0])
        predictions['expected_yield_tons'] = round(expected_yield_tons, 2)
    except Exception as e:
        print(f"Error predicting yield: {e}")
        predictions['expected_yield_tons'] = 3.5
    
    # 6. Climate risk
    try:
        climate_pred_proba = climate_model.predict_proba(X)[0]
        climate_pred_class = climate_model.predict(X)[0]
        climate_confidence = float(np.max(climate_pred_proba))
        climate_risk = climate_encoder.inverse_transform([climate_pred_class])[0]
        
        predictions['climate_risk'] = climate_risk
        predictions['climate_confidence'] = round(climate_confidence, 4)
    except Exception as e:
        print(f"Error predicting climate risk: {e}")
        predictions['climate_risk'] = 'Medium'
        predictions['climate_confidence'] = 0.0
    
    return predictions


def forecast_weather(last_7_days):
    """
    Forecast weather for the next 3 days using LSTM model.
    
    Note: LSTM weather forecasting model not yet trained.
    Currently returns None.
    
    Args:
        last_7_days (list): List of 7 dicts, each with keys: temperature, humidity, rainfall_mm, wind_speed
    
    Returns:
        None: LSTM model not yet available
    """
    if lstm_model is None:
        print("⚠ LSTM weather forecast model not yet trained")
        return None
    
    try:
        # Extract raw values and scale only first 4 features
        sequence = []
        for day_dict in last_7_days:
            raw_values = np.array([
                day_dict.get('temperature', 25.0),
                day_dict.get('humidity', 65.0),
                day_dict.get('rainfall_mm', 0.0),
                day_dict.get('wind_speed', 10.0)
            ]).reshape(1, -1)
            
            # Scale only the 4 raw features
            scaled = scaler.transform(raw_values)[:, :4]  # Only first 4 columns
            sequence.append(scaled[0])
        
        # Stack into shape (1, 7, 4)
        X_seq = np.array(sequence).reshape(1, 7, 4)
        
        # Run LSTM prediction
        y_pred = lstm_model.predict(X_seq)  # Shape: (1, 3, 4)
        
        # Inverse-transform predictions back to original scale
        forecasts = []
        for i in range(3):
            pred_4_features = y_pred[0, i, :].reshape(1, -1)
            
            # Inverse transform
            inverse_pred = scaler.inverse_transform(np.hstack([
                pred_4_features,
                np.zeros((1, 5))  # Pad with zeros for remaining 5 features
            ]))[:, :4]  # Get back only first 4
            
            temp, humid, rain, wind = inverse_pred[0]
            
            forecasts.append({
                'temperature': round(float(temp), 2),
                'humidity': round(float(humid), 2),
                'rainfall_mm': round(float(rain), 2),
                'wind_speed': round(float(wind), 2)
            })
        
        return forecasts
    
    except Exception as e:
        print(f"Error in weather forecast: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 70)
    print("AgriSense AI - Prediction Module (All 6 Models Operational)")
    print("=" * 70)
    
    # Test weather data
    test_weather = {
        'temperature': 28.5,
        'humidity': 72.0,
        'rainfall_mm': 15.0,
        'wind_speed': 12.5
    }
    
    print("\n📍 Test Weather Input:")
    print(f"   Temperature: {test_weather['temperature']}°C")
    print(f"   Humidity: {test_weather['humidity']}%")
    print(f"   Rainfall: {test_weather['rainfall_mm']}mm")
    print(f"   Wind Speed: {test_weather['wind_speed']} km/h")
    
    # Make predictions
    print("\n🔮 Running all predictions...")
    predictions = predict_all(test_weather)
    
    print("\n✓ All Predictions:")
    print(f"   Recommended Crop: {predictions['recommended_crop']} (confidence: {predictions['crop_confidence']})")
    print(f"   Disease Risk: {predictions['disease_risk']} (confidence: {predictions['disease_confidence']})")
    print(f"   Plant Stress: {predictions['plant_stress']} (confidence: {predictions['stress_confidence']})")
    print(f"   Irrigation Needed: {predictions['irrigation_need_litres']} litres")
    print(f"   Expected Yield: {predictions['expected_yield_tons']} tons/hectare")
    print(f"   Climate Risk: {predictions['climate_risk']} (confidence: {predictions['climate_confidence']})")
    
    print("\n" + "=" * 70)
