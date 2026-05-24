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
    {"name": "Karnataka","lat": 15.3, "lon": 75.1},
]

CROPS  = ["rice", "wheat", "cotton", "sugarcane"]
STAGES = ["seedling", "vegetative", "flowering", "maturity"]

# FAO-56 Table 12 crop coefficients (Kc) per stage
# Source: Allen et al. 1998, FAO-56 https://www.fao.org/4/x0490e/x0490e0b.htm
KC = {
    "rice":{"seedling": 1.05, "vegetative": 1.20, "flowering": 1.20, "maturity": 0.90},
    "wheat":{"seedling": 0.40, "vegetative": 1.15, "flowering": 1.15, "maturity": 0.30},
    "cotton":{"seedling": 0.35, "vegetative": 1.10, "flowering": 1.20, "maturity": 0.60},
    "sugarcane": {"seedling": 0.40, "vegetative": 0.85, "flowering": 1.25, "maturity": 0.75},
}

# Soil moisture trigger (%) below which irrigation is needed — loam soil assumption
# Source: FAO-56 Table 22 depletion fractions https://www.fao.org/4/x0490e/x0490e0e.htm
MOISTURE_TRIGGER = {
    "rice":{"seedling": 80, "vegetative": 80, "flowering": 80, "maturity": 65},
    "wheat":{"seedling": 22, "vegetative": 21, "flowering": 23, "maturity": 20},
    "cotton":{"seedling": 20, "vegetative": 21, "flowering": 23, "maturity": 18},
    "sugarcane":{"seedling": 22, "vegetative": 22, "flowering": 24, "maturity": 18},
}

# Water to apply per irrigation event (mm) by crop and stage
# Source: FAO-56 Table 12; ICAR-SBI Coimbatore https://sugarcane.icar.gov.in
# IRRI Knowledge Bank http://www.knowledgebank.irri.org/step-by-step-production/growth/water-management
WATER_AMOUNT = {
    "rice":{"seedling": 60,  "vegetative": 60,  "flowering": 50,  "maturity": 50},
    "wheat": {"seedling": 65,  "vegetative": 60,  "flowering": 70,  "maturity": 55},
    "cotton": {"seedling": 75,  "vegetative": 55,  "flowering": 70,  "maturity": 35},
    "sugarcane":{"seedling": 60,  "vegetative": 70,  "flowering": 78,  "maturity": 45},
}

# Max days without irrigation before crop stress  by stage
# Source: IRRI AWD guidelines; FAO-56 critical stage notes
MAX_DAYS = {
    "rice": {"seedling": 5,  "vegetative": 5,  "flowering": 3,  "maturity": 8},
    "wheat":{"seedling": 22, "vegetative": 20, "flowering": 11, "maturity": 17},
    "cotton":{"seedling": 22, "vegetative": 16, "flowering": 8,  "maturity": 22},
    "sugarcane":{"seedling": 9,  "vegetative": 11, "flowering": 8,  "maturity": 17},
}

# Rainfall in last 7 days above which irrigation is skipped
# Source: FAO Irrigation Manual Module 4 https://www.fao.org/4/S2022E/s2022e08.htm
RAIN_SKIP = {
    "rice": 30, "wheat": 20, "cotton": 25, "sugarcane": 28
}

# Reference ET0 defaults (mm/day) by season — NW India / Pakistan averages
# Source: FAO-56 climate zone tables; Frontiers in Water 2025
def get_et0(month):
    if month in [12, 1, 2]:
        return round(np.random.uniform(1.5, 3.5), 2)   # winter
    elif month in [3, 4, 5]:
        return round(np.random.uniform(5.0, 8.0), 2)   # pre-monsoon / summer
    elif month in [6, 7, 8, 9]:
        return round(np.random.uniform(3.5, 6.0), 2)   # monsoon
    else:
        return round(np.random.uniform(3.0, 5.0), 2)   # post-monsoon


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


def compute_soil_moisture(crop, stage):
    trigger = MOISTURE_TRIGGER[crop][stage]
    if crop == "rice":
        # rice is kept near saturation — simulate ponded water level as percentage
        return round(np.random.uniform(trigger - 15, 95), 1)
    return round(np.random.uniform(trigger - 10, trigger + 18), 1)


def compute_days_since_irrigation(crop, stage):
    max_d = MAX_DAYS[crop][stage]
    return int(np.random.randint(0, max_d + 5))


def decide_irrigation(soil_moisture, days_since_irrigation, rainfall_7d, crop, stage, et0):
    trigger  = MOISTURE_TRIGGER[crop][stage]
    max_days = MAX_DAYS[crop][stage]
    skip_mm  = RAIN_SKIP[crop]
    kc       = KC[crop][stage]
    etc      = round(et0 * kc, 2)

    # enough rain — skip irrigation
    if rainfall_7d >= skip_mm:
        return "no", 0, etc

    # soil still has enough moisture and not overdue
    if soil_moisture > trigger and days_since_irrigation < max_days:
        return "no", 0, etc

    # irrigate
    amount = WATER_AMOUNT[crop][stage]
    return "yes", amount, etc


all_frames = []

for loc in LOCATIONS:
    print(f"Fetching data for {loc['name']}...")
    try:
        df = fetch_nasa_weather(loc["lat"], loc["lon"], "20150101", "20251231")
        time.sleep(1)
    except Exception as e:
        print(f"  Failed: {e}")
        continue

    n = len(df)
    df["crop_type"]    = np.random.choice(CROPS,  size=n)
    df["growth_stage"] = np.random.choice(STAGES, size=n)
    df["location"]     = loc["name"]

    soil_moistures       = []
    days_since_irr       = []
    et0_vals             = []
    irrigate_labels      = []
    amount_labels        = []
    etc_vals             = []

    for _, row in df.iterrows():
        month   = int(str(row["date"])[4:6])
        crop    = row["crop_type"]
        stage   = row["growth_stage"]
        et0     = get_et0(month)
        sm      = compute_soil_moisture(crop, stage)
        days    = compute_days_since_irrigation(crop, stage)

        rain_7d = round(float(row["rainfall_mm"]) * 7, 1)

        irrigate, amount, etc = decide_irrigation(sm, days, rain_7d, crop, stage, et0)
        if irrigate == "yes" and amount > 0:
            noise = np.random.normal(0, amount * 0.10)
            amount = int(np.clip(amount + noise, amount * 0.75, amount * 1.25))

        soil_moistures.append(sm)
        days_since_irr.append(days)
        et0_vals.append(et0)
        irrigate_labels.append(irrigate)
        amount_labels.append(amount)
        etc_vals.append(etc)

    df["soil_moisture"]        = soil_moistures
    df["days_since_irrigation"] = days_since_irr
    df["et0"]                  = et0_vals
    df["etc"]                  = etc_vals
    df["irrigate"]             = irrigate_labels
    df["water_amount_mm"]      = amount_labels

    all_frames.append(df)
    print(f"  {len(df)} rows collected.")

dataset = pd.concat(all_frames, ignore_index=True)
dataset.drop(columns=["date", "location"], inplace=True)

os.makedirs("data", exist_ok=True)
dataset.to_csv("data/irrigation_dataset.csv", index=False)

print(f"\nSaved {len(dataset)} rows to data/irrigation_dataset.csv")
print("\nirrigate distribution:")
print(dataset["irrigate"].value_counts())
print("\nwater_amount_mm distribution:")
print(dataset["water_amount_mm"].value_counts())
