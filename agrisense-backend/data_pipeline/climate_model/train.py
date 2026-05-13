import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def train_climate_model():
    """
    Train a RandomForest regressor to predict crop yield from climate data.
    """
    print("=" * 70)
    print("Training Crop Yield Prediction Model")
    print("=" * 70)

    # Load dataset
    print("\n1. Loading dataset...")
    try:
        df = pd.read_csv("data/climate_dataset.csv")
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("✗ Error: data/climate_dataset.csv not found!")
        print("  Please run: python -m data_pipeline.climate_model.generate_dataset")
        return

    # Display dataset info
    print(f"\nDataset info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Define feature and target columns
    print("\n2. Preparing features and target...")
    feature_columns = [
        "temperature",
        "humidity",
        "rainfall_mm",
        "wind_speed",
        "heat_index",
        "dew_point",
        "vapor_pressure_deficit",
        "is_high_humidity",
        "is_high_temp",
    ]
    target_column = "crop_yield"

    # Check if all columns exist
    missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing_cols:
        print(f"✗ Error: Missing columns: {missing_cols}")
        return

    X = df[feature_columns]
    y = df[target_column]

    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"\nTarget (Crop Yield) Statistics:")
    print(y.describe())

    # Train-test split
    print("\n3. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Testing set: {X_test.shape[0]} samples")

    # Train RandomForest model (Regressor, not Classifier)
    print("\n4. Training RandomForestRegressor...")
    print("   (n_estimators=100, random_state=42, n_jobs=-1)")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5)

    model.fit(X_train, y_train)
    print("✓ Model training completed")

    # Make predictions
    print("\n5. Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)

    print(f"\n✓ Training Metrics:")
    print(f"   - MAE: {mae_train:.4f} tons/hectare")
    print(f"   - RMSE: {rmse_train:.4f} tons/hectare")
    print(f"   - R² Score: {r2_train:.4f}")

    print(f"\n✓ Testing Metrics:")
    print(f"   - MAE: {mae_test:.4f} tons/hectare")
    print(f"   - RMSE: {rmse_test:.4f} tons/hectare")
    print(f"   - R² Score: {r2_test:.4f}")

    # Feature importance
    print("\n" + "=" * 70)
    print("Feature Importance")
    print("=" * 70)
    feature_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(feature_importance.to_string(index=False))

    # Create models directory
    print("\n6. Saving model and artifacts...")
    os.makedirs("models", exist_ok=True)

    # Save model
    model_path = "models/climate_model.pkl"
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")

    # Save feature columns
    features_path = "models/climate_features.pkl"
    joblib.dump(feature_columns, features_path)
    print(f"✓ Feature columns saved to {features_path}")

    # Generate prediction vs actual plot
    print("\n7. Generating prediction visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted (Test set)
    axes[0].scatter(y_test, y_pred_test, alpha=0.5, color="blue")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect Prediction")
    axes[0].set_xlabel("Actual Crop Yield (tons/hectare)")
    axes[0].set_ylabel("Predicted Crop Yield (tons/hectare)")
    axes[0].set_title(f"Test Set: Actual vs Predicted (R² = {r2_test:.4f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals plot
    residuals = y_test - y_pred_test
    axes[1].scatter(y_pred_test, residuals, alpha=0.5, color="green")
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted Crop Yield (tons/hectare)")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals Plot")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = "models/crop_yield_predictions.png"
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")
    print(f"✓ Prediction visualization saved to {viz_path}")
    plt.close()

    # Train feature scaler (required for predict.py)
    print("\n8. Training feature scaler...")
    scaler_path = "models/feature_scaler.pkl"

    if not os.path.exists(scaler_path):
        feature_scaler = StandardScaler()
        feature_scaler.fit(X_train)
        joblib.dump(feature_scaler, scaler_path)
        print(f"✓ Feature scaler saved to {scaler_path}")
    else:
        print(f"✓ Feature scaler already exists at {scaler_path}")

    # Train climate risk classifier
    print("\n9. Training Climate Risk Classifier...")
    print("   (RandomForestClassifier: n_estimators=150, random_state=42, n_jobs=-1)")

    # Check if climate_risk column exists
    if "climate_risk" not in df.columns:
        print("✗ Error: 'climate_risk' column not found in dataset!")
        print("  Please ensure generate_dataset.py has been run with climate_risk computation")
        return

    # Prepare climate risk target
    y_climate = df["climate_risk"]

    # Encode climate risk labels
    climate_encoder = LabelEncoder()
    y_climate_encoded = climate_encoder.fit_transform(y_climate)

    print(f"✓ Climate risk classes: {climate_encoder.classes_}")
    print(f"✓ Climate risk distribution:\n{df['climate_risk'].value_counts()}")

    # Use same train/test split
    X_train_cr, X_test_cr, y_train_cr, y_test_cr = train_test_split(
        X, y_climate_encoded, test_size=0.2, random_state=42
    )

    # Train classifier
    climate_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)

    climate_model.fit(X_train_cr, y_train_cr)
    print("✓ Climate risk model training completed")

    # Evaluate classifier
    print("\n10. Evaluating Climate Risk Model...")
    y_pred_cr = climate_model.predict(X_test_cr)

    accuracy = accuracy_score(y_test_cr, y_pred_cr)
    print(f"\n✓ Accuracy Score: {accuracy:.4f}")
    print(f"\n✓ Classification Report:")
    print(classification_report(y_test_cr, y_pred_cr, target_names=climate_encoder.classes_))

    # Save climate risk model and encoder
    climate_model_path = "models/climate_risk_model.pkl"
    joblib.dump(climate_model, climate_model_path)
    print(f"✓ Climate risk model saved to {climate_model_path}")

    climate_encoder_path = "models/climate_risk_encoder.pkl"
    joblib.dump(climate_encoder, climate_encoder_path)
    print(f"✓ Climate risk encoder saved to {climate_encoder_path}")

    # Generate confusion matrix plot
    print("\n11. Generating confusion matrix visualization...")
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = confusion_matrix(y_test_cr, y_pred_cr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=climate_encoder.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Climate Risk Classification Confusion Matrix")
    plt.tight_layout()

    cm_path = "models/climate_risk_cm.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confusion matrix saved to {cm_path}")
    plt.close()

    print("✓ Climate risk model saved")

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\n✓ Models and encoders saved successfully")
    print(f"\n📊 Model Summary:")
    print(f"   - Crop Yield Model Type: RandomForestRegressor")
    print(f"   - Crop Yield Test R² Score: {r2_test:.4f}")
    print(f"   - Crop Yield Test MAE: {mae_test:.4f} tons/hectare")
    print(f"   - Crop Yield Test RMSE: {rmse_test:.4f} tons/hectare")
    print(f"\n   - Climate Risk Model Type: RandomForestClassifier")
    print(f"   - Climate Risk Accuracy: {accuracy:.4f}")
    print(f"   - Training samples: {X_train.shape[0]}")
    print(f"   - Testing samples: {X_test.shape[0]}")
    print(f"\n📁 Saved Files:")
    print(f"   - {model_path}")
    print(f"   - {features_path}")
    print(f"   - {scaler_path}")
    print(f"   - {climate_model_path}")
    print(f"   - {climate_encoder_path}")
    print(f"   - {viz_path}")
    print(f"   - {cm_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    train_climate_model()
