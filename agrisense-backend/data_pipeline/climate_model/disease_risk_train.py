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

def generate_disease_risk(df):
    """
    Generate disease risk labels based on climate conditions.
    Disease risk increases with humidity and moderate temperatures.
    """
    risks = []
    
    for idx, row in df.iterrows():
        temp = row['temperature']
        humidity = row['humidity']
        rainfall = row['rainfall_mm']
        
        risk_score = 0.0
        
        # High humidity increases fungal disease risk
        if humidity > 80:
            risk_score += 0.35
        elif humidity > 70:
            risk_score += 0.20
        else:
            risk_score += 0.05
        
        # Moderate temperatures (20-28°C) are ideal for disease pathogens
        if 20 <= temp <= 28:
            risk_score += 0.30
        elif 15 <= temp < 20 or 28 < temp <= 35:
            risk_score += 0.15
        else:
            risk_score += 0.02
        
        # Rainfall creates wet conditions for disease
        if rainfall > 8:
            risk_score += 0.25
        elif rainfall > 3:
            risk_score += 0.15
        else:
            risk_score += 0.05
        
        # Add some randomness
        risk_score += np.random.normal(0, 0.08)
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        # Classify
        if risk_score < 0.33:
            risk = 'Low'
        elif risk_score < 0.67:
            risk = 'Medium'
        else:
            risk = 'High'
        
        risks.append(risk)
    
    return risks

def train_disease_risk_model():
    """
    Train RandomForest classifier for disease risk prediction.
    """
    print("=" * 70)
    print("Training Disease Risk Model")
    print("=" * 70)
    
    # Load dataset
    print("\n1. Loading dataset...")
    try:
        df = pd.read_csv('data/climate_dataset.csv')
        print(f"✓ Dataset loaded: {df.shape[0]} rows")
    except FileNotFoundError:
        print("✗ Error: data/climate_dataset.csv not found!")
        return
    
    # Generate disease risk labels
    print("\n2. Generating disease risk labels...")
    df['disease_risk'] = generate_disease_risk(df)
    risk_counts = df['disease_risk'].value_counts()
    print(f"✓ Disease risk distribution:\n{risk_counts}")
    
    # Prepare features and target
    print("\n3. Preparing features and target...")
    feature_columns = [
        'temperature', 'humidity', 'rainfall_mm', 'wind_speed',
        'heat_index', 'dew_point', 'vapor_pressure_deficit',
        'is_high_humidity', 'is_high_temp'
    ]
    
    X = df[feature_columns]
    y = df['disease_risk']
    
    # Encode disease risk labels
    disease_encoder = LabelEncoder()
    y_encoded = disease_encoder.fit_transform(y)
    
    print(f"✓ Disease risk classes: {disease_encoder.classes_}")
    
    # Train-test split
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42
    )
    
    # Train model
    print("\n5. Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✓ Model training completed")
    
    # Evaluate
    print("\n6. Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Accuracy: {accuracy:.4f}")
    print(f"\n✓ Classification Report:")
    print(classification_report(y_test, y_pred, target_names=disease_encoder.classes_))
    
    # Save models
    print("\n7. Saving models...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/disease_risk_model.pkl')
    print("✓ Model saved to models/disease_risk_model.pkl")
    
    joblib.dump(disease_encoder, 'models/disease_encoder.pkl')
    print("✓ Encoder saved to models/disease_encoder.pkl")
    
    # Confusion matrix
    print("\n8. Generating confusion matrix...")
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disease_encoder.classes_)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Oranges', values_format='d')
    ax.set_title('Disease Risk Prediction Confusion Matrix')
    plt.tight_layout()
    
    plt.savefig('models/disease_risk_cm.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to models/disease_risk_cm.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✓ Disease risk model saved successfully!")
    print("=" * 70)

if __name__ == "__main__":
    train_disease_risk_model()
