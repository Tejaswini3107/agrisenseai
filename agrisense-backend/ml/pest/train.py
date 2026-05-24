import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_PATH   = "data/pest_dataset.csv"
MODELS_DIR  = "models"

FEATURE_COLS = [
    "temperature",
    "humidity",
    "rainfall_mm",
    "wind_speed",
    "crop_type",
    "growth_stage",
    "previous_pest_occurrence",
]

CATEGORICAL_COLS = ["crop_type", "growth_stage"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    return df


def encode_features(df):
    feature_encoder = OrdinalEncoder(
        categories=[
            ["rice", "wheat", "cotton", "sugarcane"],
            ["seedling", "vegetative", "flowering", "maturity"],
        ]
    )
    df = df.copy()
    df[CATEGORICAL_COLS] = feature_encoder.fit_transform(df[CATEGORICAL_COLS])
    return df, feature_encoder


def build_xy(df):
    X = df[FEATURE_COLS].values

    risk_encoder = LabelEncoder()
    y_risk = risk_encoder.fit_transform(df["pest_risk"].values)

    type_encoder = LabelEncoder()
    y_type = type_encoder.fit_transform(df["pest_type"].values)

    return X, y_risk, y_type, risk_encoder, type_encoder


def train_model(X_train, y_train, label):
    print(f"\nTraining {label} model...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, encoder, label):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"{label} — Test Accuracy: {acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    return y_pred


def save_confusion_matrix(y_test, y_pred, encoder, filename):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(filename.replace(".png", "").replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, filename))
    plt.close()
    print(f"Saved confusion matrix → models/{filename}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data()
    df, feature_encoder = encode_features(df)
    X, y_risk, y_type, risk_encoder, type_encoder = build_xy(df)

    X_train, X_test, yr_train, yr_test, yt_train, yt_test = train_test_split(
        X, y_risk, y_type,
        test_size=0.2,
        random_state=42,
        stratify=y_risk,
    )
    print(f"\nTrain size: {len(X_train)}   Test size: {len(X_test)}")

    risk_model = train_model(X_train, yr_train, "Pest Risk")
    risk_pred  = evaluate_model(risk_model, X_test, yr_test, risk_encoder, "Pest Risk")
    save_confusion_matrix(yr_test, risk_pred, risk_encoder, "pest_risk_confusion_matrix.png")

    type_model = train_model(X_train, yt_train, "Pest Type")
    type_pred  = evaluate_model(type_model, X_test, yt_test, type_encoder, "Pest Type")
    save_confusion_matrix(yt_test, type_pred, type_encoder, "pest_type_confusion_matrix.png")

    print("\nSaving model files...")
    joblib.dump(risk_model,      os.path.join(MODELS_DIR, "pest_model.pkl"))
    joblib.dump(type_model,      os.path.join(MODELS_DIR, "pest_type_model.pkl"))
    joblib.dump(risk_encoder,    os.path.join(MODELS_DIR, "pest_risk_encoder.pkl"))
    joblib.dump(type_encoder,    os.path.join(MODELS_DIR, "pest_type_encoder.pkl"))
    joblib.dump(feature_encoder, os.path.join(MODELS_DIR, "pest_feature_encoder.pkl"))

    print("\nFeature importances (Pest Risk model):")
    importances = risk_model.feature_importances_
    for col, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {col:<30} {bar}  {imp:.4f}")

    print("\nAll pest model files saved successfully.")
    print(f"Files in models/: {os.listdir(MODELS_DIR)}")


if __name__ == "__main__":
    main()
