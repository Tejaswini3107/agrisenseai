import os
import joblib
import numpy as np
import pandas as pd

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")

_clf = joblib.load(os.path.join(_MODELS_DIR, "irrigation_model.pkl"))
_reg = joblib.load(os.path.join(_MODELS_DIR, "irrigation_amount_model.pkl"))
_enc = joblib.load(os.path.join(_MODELS_DIR, "irrigation_encoder.pkl"))
_feature_encoder = joblib.load(os.path.join(_MODELS_DIR, "irrigation_feature_encoder.pkl"))

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


def predict(features: dict) -> dict:
    cat = pd.DataFrame(
        [[features["crop_type"], features["growth_stage"]]],
        columns=["crop_type", "growth_stage"]
    )
    encoded_cat = _feature_encoder.transform(cat)[0]

    kc  = features.get("kc", 1.0)
    etc = round(features["et0"] * kc, 2)

    X = np.array([[
        features["temperature"],
        features["humidity"],
        features["rainfall_mm"],
        features["soil_moisture"],
        features["days_since_irrigation"],
        features["et0"],
        etc,
        encoded_cat[0],
        encoded_cat[1],
    ]])

    irrigate_idx = _clf.predict(X)[0]
    irrigate     = str(_enc.inverse_transform([irrigate_idx])[0])

    amount = 0
    if irrigate == "yes":
        amount = int(round(_reg.predict(X)[0]))

    return {
        "irrigate":       irrigate,
        "water_amount_mm": amount,
        "etc_mm_per_day":  etc,
    }
