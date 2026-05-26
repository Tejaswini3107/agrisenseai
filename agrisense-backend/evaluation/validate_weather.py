import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# Step 1: Load the data
df = pd.read_csv('data/climate_dataset.csv')
df = df.sort_values('date', ascending=True).reset_index(drop=True)
print(f"Dataset loaded: {len(df)} rows spanning {df['date'].min()} to {df['date'].max()}")

# Step 2: Create the rain label
df['rain_label'] = (df['rainfall_mm'] > 0.5).astype(int)
rain_days = (df['rain_label'] == 1).sum()
no_rain_days = (df['rain_label'] == 0).sum()
print(f"Rain days: {rain_days}, No-rain days: {no_rain_days}")

# Step 3: Time-based 80/20 split
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()
print(f"Training period: {df_train['date'].min()} to {df_train['date'].max()}")
print(f"Test period: {df_test['date'].min()} to {df_test['date'].max()}")

# Step 4: Features for rain classifier — rainfall_mm excluded intentionally.
# The rain label is derived from rainfall_mm, so including it would let the model

RAIN_FEATURES = [
    'temperature', 'humidity', 'wind_speed',
    'heat_index', 'dew_point', 'vapor_pressure_deficit',
    'is_high_humidity', 'is_high_temp'
]

ALL_FEATURES = [
    'temperature', 'humidity', 'rainfall_mm', 'wind_speed',
    'heat_index', 'dew_point', 'vapor_pressure_deficit',
    'is_high_humidity', 'is_high_temp'
]

# Step 5: Load scaler and scale features
scaler = joblib.load('models/feature_scaler.pkl')

# Scale using all 9 features (scaler was fitted on all 9)
X_train_all = scaler.transform(df_train[ALL_FEATURES].values)
X_test_all = scaler.transform(df_test[ALL_FEATURES].values)

# For rain classifier: drop the rainfall_mm column (index 2) after scaling
rain_col_idx = ALL_FEATURES.index('rainfall_mm')
X_train_rain = np.delete(X_train_all, rain_col_idx, axis=1)
X_test_rain = np.delete(X_test_all, rain_col_idx, axis=1)

# Step 6: Train rain classifier
y_train = df_train['rain_label'].values
y_test = df_test['rain_label'].values
rain_classifier = GradientBoostingClassifier(
    n_estimators=200, random_state=42, max_depth=5,
    learning_rate=0.05, subsample=0.8
)
rain_classifier.fit(X_train_rain, y_train)
rain_proba = rain_classifier.predict_proba(X_test_rain)[:, 1]
y_pred = (rain_proba >= 0.48).astype(int)

# Step 7: Calculate rain accuracy
rain_acc = accuracy_score(y_test, y_pred)
print(f"Rain/No-Rain Accuracy: {rain_acc:.1%}  (Target: >= 75%)")

# Step 8: Temperature persistence accuracy , computed per location.
# The dataset has 7 locations. Consecutive rows in the full sorted dataset
# are from different locations, so comparing them gives meaningless results.
# Computing per-location first and then averaging gives an honest number.
location_accs = []
if 'location' in df_test.columns:
    for loc, grp in df_test.groupby('location'):
        grp = grp.sort_values('date').reset_index(drop=True)
        if len(grp) < 2:
            continue
        temps = grp['temperature'].values
        within_band = np.abs(temps[:-1] - temps[1:]) <= 2.0
        location_accs.append(within_band.mean())
    temp_acc = float(np.mean(location_accs))
else:
    # Fallback if location column is missing
    temp_values = df_test['temperature'].values
    within_band = np.abs(temp_values[:-1] - temp_values[1:]) <= 2.0
    temp_acc = within_band.mean()

print(f"Temperature +/-2C Accuracy: {temp_acc:.1%}  (Target: >= 75%)")

# Step 9: Print classification report
print("\n=== Rain Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))

# Step 10: Print summary table
rain_result = "PASS" if rain_acc >= 0.75 else "FAIL"
temp_result = "PASS" if temp_acc >= 0.75 else "FAIL"

print("-----------------------------------------------------")
print("  WEATHER FORECAST VALIDATION RESULTS")
print("-----------------------------------------------------")
print(f"  Test period : {df_test['date'].min()} to {df_test['date'].max()}")
print(f"  Test days   : {len(df_test)}")
print("-----------------------------------------------------")
print("  Metric                  Achieved   Target   Result")
print(f"  Rain/No-Rain Accuracy   {rain_acc*100:.1f}%    75.0%    {rain_result}")
print(f"  Temperature +/-2C       {temp_acc*100:.1f}%    75.0%    {temp_result}")
print("-----------------------------------------------------")

# Step 11: Save results
os.makedirs('evaluation/output', exist_ok=True)
results_df = pd.DataFrame([
    {
        'metric': 'Rain/No-Rain Accuracy',
        'achieved_pct': round(rain_acc * 100, 1),
        'target_pct': 75.0,
        'result': rain_result
    },
    {
        'metric': 'Temperature +/-2C',
        'achieved_pct': round(temp_acc * 100, 1),
        'target_pct': 75.0,
        'result': temp_result
    }
])
results_df.to_csv('evaluation/output/validation_results.csv', index=False)
print("Results saved to evaluation/output/validation_results.csv")
