import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def generate_crop_recommendations(df):
    """
    Generate synthetic crop recommendations based on climate factors.
    Using agricultural science rules for crop suitability.
    """
    crops = []
    
    for idx, row in df.iterrows():
        temp = row['temperature']
        humidity = row['humidity']
        rainfall = row['rainfall_mm']
        
        # Agricultural decision logic based on climate zones
        if 20 <= temp <= 30 and 60 <= humidity <= 75 and 2 <= rainfall <= 8:
            crop = 'Wheat'  # Cool season, moderate moisture
        elif temp > 30 and humidity > 70 and rainfall > 5:
            crop = 'Rice'   # Hot, humid, water-intensive
        elif 25 <= temp <= 35 and humidity < 65 and rainfall < 4:
            crop = 'Cotton' # Warm, dry-tolerant
        elif temp > 28 and humidity > 75 and rainfall > 8:
            crop = 'Sugarcane'  # Hot, humid, water-loving
        elif 18 <= temp <= 28 and humidity > 65 and rainfall > 3:
            crop = 'Maize'  # Moderate, moisture-loving
        elif temp < 20 and humidity < 70:
            crop = 'Barley'  # Cool, dry
        elif 25 <= temp <= 32 and humidity > 60 and rainfall > 4:
            crop = 'Groundnut'  # Warm, moderate moisture
        else:
            # Default crop selection for edge cases
            if temp > 30:
                crop = 'Rice' if humidity > 70 else 'Cotton'
            else:
                crop = 'Wheat' if humidity > 60 else 'Barley'
        
        crops.append(crop)
    
    return crops

def train_crop_recommendation_model():
    """
    Train RandomForest classifier for crop recommendation.
    """
    print("=" * 70)
    print("Training Crop Recommendation Model")
    print("=" * 70)
    
    # Load dataset
    print("\n1. Loading dataset...")
    try:
        df = pd.read_csv('data/climate_dataset.csv')
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("✗ Error: data/climate_dataset.csv not found!")
        print("  Please run: python data_pipeline/climate_model/generate_dataset.py")
        return
    
    # Generate crop recommendations
    print("\n2. Generating crop recommendation labels...")
    df['recommended_crop'] = generate_crop_recommendations(df)
    crop_counts = df['recommended_crop'].value_counts()
    print(f"✓ Crop distribution:\n{crop_counts}")
    
    # Prepare features and target
    print("\n3. Preparing features and target...")
    feature_columns = [
        'temperature', 'humidity', 'rainfall_mm', 'wind_speed',
        'heat_index', 'dew_point', 'vapor_pressure_deficit',
        'is_high_humidity', 'is_high_temp'
    ]
    
    X = df[feature_columns]
    y = df['recommended_crop']
    
    # Encode crop labels
    crop_encoder = LabelEncoder()
    y_encoded = crop_encoder.fit_transform(y)
    
    print(f"✓ Crops: {crop_encoder.classes_}")
    print(f"✓ Features shape: {X.shape}")
    
    # Train-test split
    print("\n4. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42
    )
    print(f"✓ Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
    
    # Train model
    print("\n5. Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        max_depth=15
    )
    
    model.fit(X_train, y_train)
    print("✓ Model training completed")
    
    # Evaluate
    print("\n6. Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Accuracy: {accuracy:.4f}")
    print(f"\n✓ Classification Report:")
    print(classification_report(y_test, y_pred, target_names=crop_encoder.classes_))
    
    # Save models
    print("\n7. Saving models...")
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/crop_selector.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")
    
    encoder_path = 'models/crop_label_encoder.pkl'
    joblib.dump(crop_encoder, encoder_path)
    print(f"✓ Encoder saved to {encoder_path}")
    
    # Feature importance
    print("\n8. Feature Importance:")
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.to_string(index=False))
    
    # Confusion matrix plot
    print("\n9. Generating confusion matrix...")
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=crop_encoder.classes_)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Crop Recommendation Confusion Matrix')
    plt.tight_layout()
    
    cm_path = 'models/crop_selector_cm.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {cm_path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✓ Crop recommendation model saved successfully!")
    print("=" * 70)

if __name__ == "__main__":
    train_crop_recommendation_model()
