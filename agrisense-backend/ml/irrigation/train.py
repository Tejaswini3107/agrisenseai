import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

DATA_PATH  = "data/irrigation_dataset.csv"
MODELS_DIR = "models"

FEATURE_COLS = [
    "temperature",
    "humidity",
    "rainfall_mm",
    "soil_moisture",
    "days_since_irrigation",
    "et0",
    "etc",
    "crop_type",
    "growth_stage",
]

CATEGORICAL_COLS = ["crop_type", "growth_stage"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    return df


def encode_features(df):
    encoder = OrdinalEncoder(
        categories=[
            ["rice", "wheat", "cotton", "sugarcane"],
            ["seedling", "vegetative", "flowering", "maturity"],
        ]
    )
    df = df.copy()
    df[CATEGORICAL_COLS] = encoder.fit_transform(df[CATEGORICAL_COLS])
    return df, encoder


def build_xy(df):
    X = df[FEATURE_COLS].values

    irrigate_encoder = LabelEncoder()
    y_irrigate = irrigate_encoder.fit_transform(df["irrigate"].values)

    y_amount = df["water_amount_mm"].values

    return X, y_irrigate, y_amount, irrigate_encoder


def train_classifier(X_train, y_train):
    print("\nTraining irrigation decision model...")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def train_regressor(X_train, y_train):
    print("\nTraining water amount model...")
    # only train on rows where irrigation was needed
    mask = y_train > 0
    model = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train[mask], y_train[mask])
    return model


def evaluate_classifier(model, X_test, y_test, encoder):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"Irrigation Decision — Test Accuracy: {acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    return y_pred


def evaluate_regressor(model, X_test, y_test):
    mask = y_test > 0
    y_pred = model.predict(X_test[mask])
    mae = mean_absolute_error(y_test[mask], y_pred)
    r2  = r2_score(y_test[mask], y_pred)
    print(f"\n{'='*60}")
    print(f"Water Amount Model — MAE: {mae:.2f} mm   R2: {r2:.4f}")
    print(f"{'='*60}")
    return y_pred


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data()
    df, feature_encoder = encode_features(df)
    X, y_irrigate, y_amount, irrigate_encoder = build_xy(df)

    X_train, X_test, yi_train, yi_test, ya_train, ya_test = train_test_split(
        X, y_irrigate, y_amount,
        test_size=0.2,
        random_state=42,
        stratify=y_irrigate,
    )
    print(f"\nTrain size: {len(X_train)}   Test size: {len(X_test)}")

    clf = train_classifier(X_train, yi_train)
    evaluate_classifier(clf, X_test, yi_test, irrigate_encoder)

    reg = train_regressor(X_train, ya_train)
    evaluate_regressor(reg, X_test, ya_test)

    print("\nSaving model files...")
    joblib.dump(clf,             os.path.join(MODELS_DIR, "irrigation_model.pkl"))
    joblib.dump(reg,             os.path.join(MODELS_DIR, "irrigation_amount_model.pkl"))
    joblib.dump(irrigate_encoder, os.path.join(MODELS_DIR, "irrigation_encoder.pkl"))
    joblib.dump(feature_encoder,  os.path.join(MODELS_DIR, "irrigation_feature_encoder.pkl"))

    print("\nFeature importances (irrigation decision model):")
    for col, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {col:<28} {bar}  {imp:.4f}")

    print("\nAll irrigation model files saved.")
    print(f"Files in models/: {os.listdir(MODELS_DIR)}")


if __name__ == "__main__":
    main()
