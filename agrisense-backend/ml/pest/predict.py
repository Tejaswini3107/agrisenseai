import os
import joblib
import numpy as np
import pandas as pd

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")

_risk_model      = joblib.load(os.path.join(_MODELS_DIR, "pest_model.pkl"))
_type_model      = joblib.load(os.path.join(_MODELS_DIR, "pest_type_model.pkl"))
_risk_encoder    = joblib.load(os.path.join(_MODELS_DIR, "pest_risk_encoder.pkl"))
_type_encoder    = joblib.load(os.path.join(_MODELS_DIR, "pest_type_encoder.pkl"))
_feature_encoder = joblib.load(os.path.join(_MODELS_DIR, "pest_feature_encoder.pkl"))


def predict(features: dict) -> dict:
    cat = pd.DataFrame([[features["crop_type"], features["growth_stage"]]], columns=["crop_type", "growth_stage"])
    encoded_cat = _feature_encoder.transform(cat)[0]

    X = np.array([[
        features["temperature"],
        features["humidity"],
        features["rainfall_mm"],
        features["wind_speed"],
        encoded_cat[0],
        encoded_cat[1],
        features["previous_pest_occurrence"],
    ]])

    risk_idx    = _risk_model.predict(X)[0]
    risk_proba  = _risk_model.predict_proba(X)[0]
    confidence  = float(np.max(risk_proba))
    risk_level  = str(_risk_encoder.inverse_transform([risk_idx])[0])

    type_idx  = _type_model.predict(X)[0]
    pest_type = str(_type_encoder.inverse_transform([type_idx])[0])

    return {
        "risk_level": risk_level,
        "pest_type":  pest_type,
        "confidence": round(confidence, 4),
    }
