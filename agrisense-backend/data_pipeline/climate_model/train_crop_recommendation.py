import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

def train_crop_recommendation_models():
    """
    Train integrated models for crop recommendation system.
    Includes: crop suitability, disease risk, irrigation, stress, and yield prediction.
    """
    print("=" * 70)
    print("Training Crop Recommendation System")
    print("=" * 70)
    
    # Load dataset
    print("\n1. Loading crop recommendation dataset...")
    try:
        df = pd.read_csv('data/crop_recommendation_dataset.csv')
        print(f"✓ Loaded {len(df)} records")
    except FileNotFoundError:
        print("✗ Dataset not found! Running generation first...")
        os.system('python -m data_pipeline.climate_model.generate_crop_recommendation_dataset')
        df = pd.read_csv('data/crop_recommendation_dataset.csv')
    
    # Feature columns
    feature_columns = [
        'temperature', 'humidity', 'rainfall_mm', 'wind_speed',
        'heat_index', 'dew_point', 'vapor_pressure_deficit',
        'is_high_humidity', 'is_high_temp'
    ]
    
    X = df[feature_columns]
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # ==================== Model 1: Crop Suitability Classifier ====================
    print("\n2. Training Crop Suitability Classifier...")
    
    y_crop = df['best_crop']
    crop_encoder = LabelEncoder()
    y_crop_encoded = crop_encoder.fit_transform(y_crop)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_crop_encoded, test_size=0.2, random_state=42, stratify=y_crop_encoded
    )
    
    crop_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    crop_model.fit(X_train, y_train)
    
    crop_accuracy = accuracy_score(y_test, crop_model.predict(X_test))
    print(f"✓ Crop Suitability Accuracy: {crop_accuracy:.4f} ({crop_accuracy*100:.2f}%)")
    print("\nCrop Prediction Report:")
    print(classification_report(y_test, crop_model.predict(X_test), 
                              target_names=crop_encoder.classes_))
    
    joblib.dump(crop_model, 'models/crop_selector_model.pkl')
    joblib.dump(crop_encoder, 'models/crop_encoder.pkl')
    print("✓ Crop model saved")
    
    # ==================== Model 2: Disease Risk Classifier ====================
    print("\n3. Training Disease Risk Classifier...")
    
    y_disease = df['disease_risk']
    disease_encoder = LabelEncoder()
    y_disease_encoded = disease_encoder.fit_transform(y_disease)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_disease_encoded, test_size=0.2, random_state=42, stratify=y_disease_encoded
    )
    
    disease_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    disease_model.fit(X_train, y_train)
    
    disease_accuracy = accuracy_score(y_test, disease_model.predict(X_test))
    print(f"✓ Disease Risk Accuracy: {disease_accuracy:.4f} ({disease_accuracy*100:.2f}%)")
    
    joblib.dump(disease_model, 'models/disease_risk_model.pkl')
    joblib.dump(disease_encoder, 'models/disease_encoder.pkl')
    print("✓ Disease model saved")
    
    # ==================== Model 3: Plant Stress Classifier ====================
    print("\n4. Training Plant Stress Classifier...")
    
    y_stress = df['plant_stress']
    stress_encoder = LabelEncoder()
    y_stress_encoded = stress_encoder.fit_transform(y_stress)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_stress_encoded, test_size=0.2, random_state=42, stratify=y_stress_encoded
    )
    
    stress_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    stress_model.fit(X_train, y_train)
    
    stress_accuracy = accuracy_score(y_test, stress_model.predict(X_test))
    print(f"✓ Plant Stress Accuracy: {stress_accuracy:.4f} ({stress_accuracy*100:.2f}%)")
    
    joblib.dump(stress_model, 'models/plant_stress_model.pkl')
    joblib.dump(stress_encoder, 'models/stress_encoder.pkl')
    print("✓ Stress model saved")
    
    # ==================== Model 4: Irrigation Requirement Regressor ====================
    print("\n5. Training Irrigation Requirement Regressor...")
    
    y_irrigation = df['irrigation_need']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_irrigation, test_size=0.2, random_state=42
    )
    
    irrigation_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    irrigation_model.fit(X_train, y_train)
    
    y_pred = irrigation_model.predict(X_test)
    irrigation_mae = mean_absolute_error(y_test, y_pred)
    irrigation_r2 = r2_score(y_test, y_pred)
    print(f"✓ Irrigation MAE: {irrigation_mae:.4f} L/m²")
    print(f"✓ Irrigation R² Score: {irrigation_r2:.4f}")
    
    joblib.dump(irrigation_model, 'models/irrigation_requirement_model.pkl')
    print("✓ Irrigation model saved")
    
    # ==================== Model 5: Yield Prediction Regressor ====================
    print("\n6. Training Yield Prediction Regressor...")
    
    y_yield = df['expected_yield']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_yield, test_size=0.2, random_state=42
    )
    
    yield_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
    yield_model.fit(X_train, y_train)
    
    y_pred = yield_model.predict(X_test)
    yield_mae = mean_absolute_error(y_test, y_pred)
    yield_r2 = r2_score(y_test, y_pred)
    print(f"✓ Yield MAE: {yield_mae:.4f} tons/hectare")
    print(f"✓ Yield R² Score: {yield_r2:.4f}")
    
    joblib.dump(yield_model, 'models/yield_prediction_model.pkl')
    print("✓ Yield model saved")
    
    # ==================== Feature Importance Analysis ====================
    print("\n7. Feature Importance Across Models:")
    print("-" * 70)
    
    feature_importances = pd.DataFrame({
        'Crop': crop_model.feature_importances_,
        'Disease': disease_model.feature_importances_,
        'Stress': stress_model.feature_importances_,
        'Irrigation': irrigation_model.feature_importances_,
        'Yield': yield_model.feature_importances_
    }, index=feature_columns)
    
    feature_importances['Mean'] = feature_importances.mean(axis=1)
    feature_importances = feature_importances.sort_values('Mean', ascending=False)
    
    print(feature_importances)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    models_info = [
        ('Crop Selector', crop_model.feature_importances_),
        ('Disease Risk', disease_model.feature_importances_),
        ('Plant Stress', stress_model.feature_importances_),
        ('Irrigation', irrigation_model.feature_importances_),
        ('Yield', yield_model.feature_importances_),
        ('Mean Importance', feature_importances['Mean'].values)
    ]
    
    for idx, (title, importances) in enumerate(models_info):
        sorted_idx = np.argsort(importances)[::-1][:9]
        sorted_features = [feature_columns[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]
        
        axes[idx].barh(sorted_features, sorted_importances, color='steelblue')
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('models/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved")
    plt.close()
    
    # ==================== Final Summary ====================
    print("\n" + "=" * 70)
    print("Crop Recommendation System Training Complete!")
    print("=" * 70)
    
    summary = f"""
📊 Model Performance Summary:
   
   1. Crop Suitability Classifier
      - Accuracy: {crop_accuracy*100:.2f}%
      - Crops: {', '.join(crop_encoder.classes_)}
   
   2. Disease Risk Classifier
      - Accuracy: {disease_accuracy*100:.2f}%
      - Risk Levels: Low, Medium, High
   
   3. Plant Stress Classifier
      - Accuracy: {stress_accuracy*100:.2f}%
      - Stress Levels: Low, Medium, High
   
   4. Irrigation Requirement Regressor
      - MAE: {irrigation_mae:.4f} L/m²
      - R² Score: {irrigation_r2:.4f}
   
   5. Yield Prediction Regressor
      - MAE: {yield_mae:.4f} tons/hectare
      - R² Score: {yield_r2:.4f}

📁 Saved Models:
   - models/crop_selector_model.pkl
   - models/disease_risk_model.pkl
   - models/plant_stress_model.pkl
   - models/irrigation_requirement_model.pkl
   - models/yield_prediction_model.pkl
   - models/feature_importance_analysis.png

🎯 Top 3 Most Important Features:
   1. {feature_importances.index[0]} ({feature_importances['Mean'].iloc[0]:.4f})
   2. {feature_importances.index[1]} ({feature_importances['Mean'].iloc[1]:.4f})
   3. {feature_importances.index[2]} ({feature_importances['Mean'].iloc[2]:.4f})
"""
    
    print(summary)
    print("=" * 70)


if __name__ == "__main__":
    train_crop_recommendation_models()
