import os
import time
import requests
import numpy as np
import pandas as pd

np.random.seed(42)

LOCATIONS = [
    {"name": "Punjab","lat": 30.9, "lon": 75.8},
    {"name": "Maharashtra","lat": 19.0, "lon": 76.1},
    {"name": "Andhra Pradesh","lat": 15.9, "lon": 79.7},
    {"name": "Tamil Nadu","lat": 11.1, "lon": 77.3},
    {"name": "Uttar Pradesh","lat": 27.0, "lon": 80.9},
    {"name": "Faisalabad","lat": 31.4, "lon": 73.1},
    {"name": "Rahim Yar Khan","lat": 28.4, "lon": 70.3},
    {"name": "Nizamabad","lat": 18.6, "lon": 78.1},
    {"name": "karnataka","lat": 15.3, "lon": 75.1},

]

CROPS = ["rice", "wheat", "cotton", "tomato"]
STAGES = ["seedling", "vegetative", "flowering", "maturity"]

PEST_BY_CROP = {
    "rice":   ["stem_borer","leaf_miner","aphids"],
    "wheat":  ["aphids","stem_borer", "leaf_miner"],
    "cotton": ["bollworm","whitefly","aphids"],
    "tomato": ["whitefly","aphids","bollworm"],
}

STAGE_RISK_WEIGHT = {
    "seedling":   0.05,
    "vegetative": 0.15,
    "flowering":  0.25,
    "maturity":   0.10,
}


def fetch_nasa_weather(lat, lon, start, end):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR,WS2M",
        "community":  "AG",
        "longitude":  lon,
        "latitude":   lat,
        "start":      start,
        "end":        end,
        "format":     "JSON",
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()["properties"]["parameter"]

    dates        = list(data["T2M"].keys())
    temperatures = list(data["T2M"].values())
    humidities   = list(data["RH2M"].values())
    rainfalls    = list(data["PRECTOTCORR"].values())
    # NASA gives wind in metres per second. Multiply by 3.6 to get km/h.
    wind_speeds  = [v * 3.6 for v in data["WS2M"].values()]

    df = pd.DataFrame({
        "date":        dates,
        "temperature": temperatures,
        "humidity":    humidities,
        "rainfall_mm": rainfalls,
        "wind_speed":  wind_speeds,
    })

    df.replace(-999.0, np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_pest_risk(temperature, humidity, rainfall_mm, wind_speed, crop, stage, prev):
    score = 0.0

    if temperature > 30:
        score += 0.25
    elif temperature > 25:
        score += 0.15
    else:
        score += 0.05

    if humidity > 75:
        score += 0.25
    elif humidity > 55:
        score += 0.15
    else:
        score += 0.05

    if 20 <= rainfall_mm <= 80:
        score += 0.15
    elif rainfall_mm > 80:
        score += 0.05
    else:
        score += 0.08

    if wind_speed < 10:
        score += 0.10

    score += STAGE_RISK_WEIGHT[stage]

    if prev == 1:
        score *= 1.5

    # adding some noise to make the dataset more realistic

    noise = np.random.normal(0, 0.04)
    score = float(np.clip(score + noise, 0.0, 1.0))
    return score


def label_pest(score, crop):
    if score < 0.50:
        return "low", "none"
    elif score < 0.73:
        return "medium", np.random.choice(PEST_BY_CROP[crop])
    else:
        return "high", PEST_BY_CROP[crop][0]


all_frames = []

for loc in LOCATIONS:
    print(f"Fetching data for {loc['name']}...")
    try:
        df = fetch_nasa_weather(loc["lat"], loc["lon"], "20190101", "20231231")
        time.sleep(1)
    except Exception as e:
        print(f"  Failed: {e}")
        continue

    n = len(df)
    df["crop_type"] = np.random.choice(CROPS,   size=n)
    df["growth_stage"] = np.random.choice(STAGES,  size=n)
    df["previous_pest_occurrence"] = np.random.choice([0, 1], size=n, p=[0.65, 0.35])
    df["location"]  = loc["name"]

    pest_risks  = []
    pest_types  = []

    for _, row in df.iterrows():
        score = compute_pest_risk(
            row["temperature"],
            row["humidity"],
            row["rainfall_mm"],
            row["wind_speed"],
            row["crop_type"],
            row["growth_stage"],
            row["previous_pest_occurrence"],
        )
        risk, pest = label_pest(score, row["crop_type"])
        pest_risks.append(risk)
        pest_types.append(pest)

    df["pest_risk"] = pest_risks
    df["pest_type"] = pest_types

    all_frames.append(df)
    print(f"  {len(df)} rows collected.")

dataset = pd.concat(all_frames, ignore_index=True)
dataset.drop(columns=["date", "location"], inplace=True)

os.makedirs("data", exist_ok=True)
dataset.to_csv("data/pest_dataset.csv", index=False)

print(f"\nSaved {len(dataset)} rows to data/pest_dataset.csv")
print("\npest_risk distribution:")
print(dataset["pest_risk"].value_counts())
print("\npest_type distribution:")
print(dataset["pest_type"].value_counts())
