import numpy as np
import joblib

def get_crop_recommendation(weather_features):
    """
    Get integrated crop recommendation based on current weather.
    
    Args:
        weather_features: Dict with keys: temperature, humidity, rainfall_mm, wind_speed,
                         heat_index, dew_point, vapor_pressure_deficit, 
                         is_high_humidity, is_high_temp
    
    Returns:
        Dict with complete recommendation including crop, disease risk, irrigation, stress, yield
    """
    print("=" * 70)
    print("🌾 Crop Recommendation System")
    print("=" * 70)
    
    # Load all models
    print("\n📦 Loading models...")
    try:
        crop_model = joblib.load('models/crop_selector_model.pkl')
        crop_encoder = joblib.load('models/crop_encoder.pkl')
        disease_model = joblib.load('models/disease_risk_model.pkl')
        disease_encoder = joblib.load('models/disease_encoder.pkl')
        stress_model = joblib.load('models/plant_stress_model.pkl')
        stress_encoder = joblib.load('models/stress_encoder.pkl')
        irrigation_model = joblib.load('models/irrigation_requirement_model.pkl')
        yield_model = joblib.load('models/yield_prediction_model.pkl')
        crop_requirements = joblib.load('data/crop_requirements.pkl')
        
        print("✓ All models loaded successfully")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Please run: python -m data_pipeline.climate_model.train_crop_recommendation")
        return None
    
    # Prepare features
    print("\n🌡️  Weather Input:")
    feature_names = [
        'temperature', 'humidity', 'rainfall_mm', 'wind_speed',
        'heat_index', 'dew_point', 'vapor_pressure_deficit',
        'is_high_humidity', 'is_high_temp'
    ]
    
    X = np.array([weather_features[f] for f in feature_names]).reshape(1, -1)
    
    print(f"   Temperature: {weather_features['temperature']:.2f}°C")
    print(f"   Humidity: {weather_features['humidity']:.2f}%")
    print(f"   Rainfall: {weather_features['rainfall_mm']:.2f}mm")
    print(f"   Wind Speed: {weather_features['wind_speed']:.2f}km/h")
    print(f"   VPD: {weather_features['vapor_pressure_deficit']:.4f}")
    
    # Make predictions
    print("\n🤖 Generating recommendations...")
    
    # Crop prediction
    crop_pred = crop_model.predict(X)[0]
    crop_proba = crop_model.predict_proba(X)[0]
    best_crop = crop_encoder.classes_[crop_pred]
    crop_confidence = crop_proba[crop_pred]
    
    # Get top 3 crops
    top_3_idx = np.argsort(crop_proba)[::-1][:3]
    top_3_crops = [(crop_encoder.classes_[i], crop_proba[i]) for i in top_3_idx]
    
    # Disease risk prediction
    disease_pred = disease_model.predict(X)[0]
    disease_proba = disease_model.predict_proba(X)[0]
    disease_risk = disease_encoder.classes_[disease_pred]
    disease_confidence = disease_proba[disease_pred]
    
    # Plant stress prediction
    stress_pred = stress_model.predict(X)[0]
    stress_proba = stress_model.predict_proba(X)[0]
    plant_stress = stress_encoder.classes_[stress_pred]
    stress_confidence = stress_proba[stress_pred]
    
    # Irrigation requirement
    irrigation_need = irrigation_model.predict(X)[0]
    
    # Yield prediction
    expected_yield = yield_model.predict(X)[0]
    
    # Format output
    print("\n" + "=" * 70)
    print("💡 CROP RECOMMENDATION RESULTS")
    print("=" * 70)
    
    print(f"\n🌾 PRIMARY RECOMMENDATION: {best_crop}")
    print(f"   Confidence Score: {crop_confidence*100:.1f}%")
    print(f"   Suitability: {'Excellent' if crop_confidence > 0.8 else 'Good' if crop_confidence > 0.6 else 'Fair'}")
    
    print(f"\n🏆 Top 3 Recommended Crops:")
    for i, (crop, prob) in enumerate(top_3_crops, 1):
        print(f"   {i}. {crop} ({prob*100:.1f}% suitable)")
    
    print(f"\n⚠️  DISEASE RISK: {disease_risk}")
    print(f"   Risk Level: {disease_confidence*100:.1f}%")
    if disease_risk == 'High':
        print("   ⚠️  Recommendation: Monitor crop closely, consider preventive spraying")
    elif disease_risk == 'Medium':
        print("   ⚠️  Recommendation: Maintain good farm hygiene, watch for symptoms")
    else:
        print("   ✓ Low risk, continue normal practices")
    
    print(f"\n😰 PLANT STRESS: {plant_stress}")
    print(f"   Stress Level: {stress_confidence*100:.1f}%")
    if plant_stress == 'High':
        print("   🚨 ACTION NEEDED: Immediate irrigation or stress management required")
    elif plant_stress == 'Medium':
        print("   ⚠️  Recommendation: Monitor water availability, may need irrigation soon")
    else:
        print("   ✓ Crop is thriving, minimal stress")
    
    print(f"\n💧 IRRIGATION REQUIREMENT:")
    print(f"   {irrigation_need:.2f} liters/m² today")
    print(f"   (Based on VPD and rainfall)")
    
    print(f"\n📊 EXPECTED YIELD: {expected_yield:.2f} tons/hectare")
    print(f"   (Based on current conditions)")
    
    # Suitability factors
    print(f"\n✅ CROP-SPECIFIC ANALYSIS FOR {best_crop}:")
    reqs = crop_requirements[best_crop]
    
    factors = []
    
    # Temperature
    if reqs['temp_min'] <= weather_features['temperature'] <= reqs['temp_max']:
        factors.append(f"   ✓ Temperature {weather_features['temperature']:.1f}°C (Optimal: {reqs['temp_min']}-{reqs['temp_max']}°C)")
    else:
        factors.append(f"   ✗ Temperature {weather_features['temperature']:.1f}°C (Optimal: {reqs['temp_min']}-{reqs['temp_max']}°C)")
    
    # Humidity
    if reqs['humidity_min'] <= weather_features['humidity'] <= reqs['humidity_max']:
        factors.append(f"   ✓ Humidity {weather_features['humidity']:.1f}% (Optimal: {reqs['humidity_min']}-{reqs['humidity_max']}%)")
    else:
        factors.append(f"   ✗ Humidity {weather_features['humidity']:.1f}% (Optimal: {reqs['humidity_min']}-{reqs['humidity_max']}%)")
    
    # Rainfall
    if reqs['rainfall_min'] <= weather_features['rainfall_mm'] <= reqs['rainfall_max']:
        factors.append(f"   ✓ Rainfall {weather_features['rainfall_mm']:.2f}mm (Optimal: {reqs['rainfall_min']}-{reqs['rainfall_max']}mm)")
    else:
        factors.append(f"   ✗ Rainfall {weather_features['rainfall_mm']:.2f}mm (Optimal: {reqs['rainfall_min']}-{reqs['rainfall_max']}mm)")
    
    # VPD
    if reqs['vpd_min'] <= weather_features['vapor_pressure_deficit'] <= reqs['vpd_max']:
        factors.append(f"   ✓ VPD {weather_features['vapor_pressure_deficit']:.2f} (Optimal: {reqs['vpd_min']}-{reqs['vpd_max']})")
    else:
        factors.append(f"   ✗ VPD {weather_features['vapor_pressure_deficit']:.2f} (Optimal: {reqs['vpd_min']}-{reqs['vpd_max']})")
    
    for factor in factors:
        print(factor)
    
    print("\n" + "=" * 70)
    
    # Return recommendation dict
    recommendation = {
        'best_crop': best_crop,
        'crop_confidence': float(crop_confidence),
        'top_3_crops': [(crop, float(prob)) for crop, prob in top_3_crops],
        'disease_risk': disease_risk,
        'disease_confidence': float(disease_confidence),
        'plant_stress': plant_stress,
        'stress_confidence': float(stress_confidence),
        'irrigation_need_lpm2': float(irrigation_need),
        'expected_yield_tons': float(expected_yield)
    }
    
    return recommendation


if __name__ == "__main__":
    # Example: typical weather conditions in Punjab
    example_weather = {
        'temperature': 27.5,
        'humidity': 65.0,
        'rainfall_mm': 3.5,
        'wind_speed': 8.5,
        'heat_index': 30.2,
        'dew_point': 18.5,
        'vapor_pressure_deficit': 1.8,
        'is_high_humidity': 0,
        'is_high_temp': 0
    }
    
    print("Example: Current weather conditions\n")
    recommendation = get_crop_recommendation(example_weather)
    
    if recommendation:
        print("\n✅ Recommendation generated successfully!")
        print(f"\nJSON Output:")
        import json
        print(json.dumps(recommendation, indent=2))
