import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

def generate_crop_recommendation_dataset():
    """
    Generate synthetic crop recommendation dataset based on climate patterns
    and crop requirements across 5 Indian regions.
    """
    print("=" * 70)
    print("Generating Crop Recommendation Dataset")
    print("=" * 70)
    
    # Load existing climate data
    print("\n1. Loading climate dataset...")
    df = pd.read_csv('data/climate_dataset.csv')
    print(f"✓ Loaded {len(df)} climate records")
    
    # Crop requirements (temp, humidity, rainfall ranges)
    # Format: (temp_min, temp_max, humidity_min, humidity_max, rainfall_min_daily, rainfall_max_daily, vpd_min, vpd_max)
    CROP_REQUIREMENTS = {
        'Rice': {
            'temp_min': 20, 'temp_max': 32, 
            'humidity_min': 60, 'humidity_max': 95,
            'rainfall_min': 1.5, 'rainfall_max': 100,
            'vpd_min': 0.3, 'vpd_max': 3.0,
            'disease_risk_humidity': 0.85,
            'irrigation_base': 4.0
        },
        'Wheat': {
            'temp_min': 15, 'temp_max': 28,
            'humidity_min': 40, 'humidity_max': 80,
            'rainfall_min': 0.5, 'rainfall_max': 15,
            'vpd_min': 0.5, 'vpd_max': 4.0,
            'disease_risk_humidity': 0.75,
            'irrigation_base': 2.5
        },
        'Maize': {
            'temp_min': 18, 'temp_max': 30,
            'humidity_min': 50, 'humidity_max': 85,
            'rainfall_min': 2.0, 'rainfall_max': 50,
            'vpd_min': 0.4, 'vpd_max': 3.5,
            'disease_risk_humidity': 0.80,
            'irrigation_base': 3.5
        },
        'Cotton': {
            'temp_min': 22, 'temp_max': 35,
            'humidity_min': 30, 'humidity_max': 70,
            'rainfall_min': 0.5, 'rainfall_max': 20,
            'vpd_min': 1.0, 'vpd_max': 5.0,
            'disease_risk_humidity': 0.65,
            'irrigation_base': 2.0
        },
        'Sugarcane': {
            'temp_min': 21, 'temp_max': 33,
            'humidity_min': 60, 'humidity_max': 90,
            'rainfall_min': 2.0, 'rainfall_max': 80,
            'vpd_min': 0.4, 'vpd_max': 3.5,
            'disease_risk_humidity': 0.88,
            'irrigation_base': 5.0
        }
    }
    
    print("\n2. Creating crop suitability labels...")
    
    def get_crop_suitability(row):
        """Determine best crop and suitability score."""
        scores = {}
        
        for crop, reqs in CROP_REQUIREMENTS.items():
            score = 100
            
            # Temperature score
            if reqs['temp_min'] <= row['temperature'] <= reqs['temp_max']:
                temp_score = 100
            else:
                # Penalty for being outside range
                if row['temperature'] < reqs['temp_min']:
                    temp_score = max(0, 100 - 5 * (reqs['temp_min'] - row['temperature']))
                else:
                    temp_score = max(0, 100 - 5 * (row['temperature'] - reqs['temp_max']))
            
            # Humidity score
            if reqs['humidity_min'] <= row['humidity'] <= reqs['humidity_max']:
                humidity_score = 100
            else:
                if row['humidity'] < reqs['humidity_min']:
                    humidity_score = max(0, 100 - 3 * (reqs['humidity_min'] - row['humidity']))
                else:
                    humidity_score = max(0, 100 - 3 * (row['humidity'] - reqs['humidity_max']))
            
            # Rainfall score
            if reqs['rainfall_min'] <= row['rainfall_mm'] <= reqs['rainfall_max']:
                rainfall_score = 100
            else:
                if row['rainfall_mm'] < reqs['rainfall_min']:
                    rainfall_score = max(0, 100 - 2 * (reqs['rainfall_min'] - row['rainfall_mm']))
                else:
                    rainfall_score = max(0, 100 - 2 * (row['rainfall_mm'] - reqs['rainfall_max']))
            
            # VPD score
            if reqs['vpd_min'] <= row['vapor_pressure_deficit'] <= reqs['vpd_max']:
                vpd_score = 100
            else:
                if row['vapor_pressure_deficit'] < reqs['vpd_min']:
                    vpd_score = max(0, 100 - 5 * (reqs['vpd_min'] - row['vapor_pressure_deficit']))
                else:
                    vpd_score = max(0, 100 - 5 * (row['vapor_pressure_deficit'] - reqs['vpd_max']))
            
            # Combine scores (weights)
            combined_score = (temp_score * 0.3 + humidity_score * 0.25 + 
                            rainfall_score * 0.25 + vpd_score * 0.2)
            scores[crop] = combined_score
        
        best_crop = max(scores, key=scores.get)
        best_score = scores[best_crop] / 100  # Normalize to 0-1
        
        return best_crop, best_score
    
    # Apply crop suitability
    results = df.apply(lambda row: get_crop_suitability(row), axis=1, result_type='expand')
    df['best_crop'] = results[0]
    df['crop_suitability_score'] = results[1]
    
    print("\n3. Calculating crop-specific metrics...")
    
    def calculate_disease_risk(row, crop):
        """Calculate disease risk based on humidity and temperature."""
        reqs = CROP_REQUIREMENTS[crop]
        
        base_risk = 0.3
        
        # High humidity increases disease risk
        humidity_factor = (row['humidity'] / 100) ** 2 if row['humidity'] > reqs['disease_risk_humidity'] * 100 else 0
        
        # Moderate temperature increases disease
        if 20 <= row['temperature'] <= 28:
            temp_factor = 0.3
        else:
            temp_factor = 0.1
        
        disease_risk = min(0.95, base_risk + humidity_factor * 0.4 + temp_factor * 0.3)
        return disease_risk
    
    def calculate_irrigation_need(row, crop):
        """Calculate irrigation need based on VPD."""
        reqs = CROP_REQUIREMENTS[crop]
        
        # VPD directly correlates with water stress
        vpd = row['vapor_pressure_deficit']
        base_irrigation = reqs['irrigation_base']
        
        # High VPD = more water needed
        vpd_factor = (vpd / 4.0) if vpd > 2.0 else 0
        rainfall_reduction = row['rainfall_mm'] / 10
        
        irrigation = max(0.5, base_irrigation + vpd_factor - rainfall_reduction)
        return irrigation
    
    def calculate_plant_stress(row, crop):
        """Calculate plant stress level."""
        reqs = CROP_REQUIREMENTS[crop]
        
        stress = 0
        
        # Temperature stress
        if row['temperature'] > reqs['temp_max']:
            stress += (row['temperature'] - reqs['temp_max']) * 0.3
        elif row['temperature'] < reqs['temp_min']:
            stress += (reqs['temp_min'] - row['temperature']) * 0.3
        
        # VPD stress (high VPD = water stress)
        if row['vapor_pressure_deficit'] > reqs['vpd_max']:
            stress += (row['vapor_pressure_deficit'] - reqs['vpd_max']) * 0.4
        
        # Humidity stress
        if row['humidity'] > reqs['humidity_max']:
            stress += (row['humidity'] - reqs['humidity_max']) * 0.2
        elif row['humidity'] < reqs['humidity_min']:
            stress += (reqs['humidity_min'] - row['humidity']) * 0.2
        
        # Categorize
        if stress < 2:
            return 'Low'
        elif stress < 5:
            return 'Medium'
        else:
            return 'High'
    
    df['disease_risk_numeric'] = df.apply(
        lambda row: calculate_disease_risk(row, row['best_crop']), axis=1
    )
    df['irrigation_need'] = df.apply(
        lambda row: calculate_irrigation_need(row, row['best_crop']), axis=1
    )
    df['plant_stress'] = df.apply(
        lambda row: calculate_plant_stress(row, row['best_crop']), axis=1
    )
    
    # Disease category
    def categorize_disease_risk(risk):
        if risk < 0.35:
            return 'Low'
        elif risk < 0.65:
            return 'Medium'
        else:
            return 'High'
    
    df['disease_risk'] = df['disease_risk_numeric'].apply(categorize_disease_risk)
    
    # Yield prediction based on suitability
    df['expected_yield'] = df['crop_suitability_score'] * 5.0 + np.random.normal(0, 0.3, len(df))
    df['expected_yield'] = df['expected_yield'].clip(lower=0.5)
    
    print("\n4. Final dataset statistics:")
    print(f"   Best crops distribution:")
    print(df['best_crop'].value_counts())
    print(f"\n   Disease risk distribution:")
    print(df['disease_risk'].value_counts())
    print(f"\n   Plant stress distribution:")
    print(df['plant_stress'].value_counts())
    
    # Save dataset
    print("\n5. Saving crop recommendation dataset...")
    os.makedirs('data', exist_ok=True)
    
    dataset_path = 'data/crop_recommendation_dataset.csv'
    df.to_csv(dataset_path, index=False)
    print(f"✓ Dataset saved to {dataset_path}")
    
    # Save crop requirements
    joblib.dump(CROP_REQUIREMENTS, 'data/crop_requirements.pkl')
    print(f"✓ Crop requirements saved")
    
    print("\n" + "=" * 70)
    print("Dataset generation completed!")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    generate_crop_recommendation_dataset()
