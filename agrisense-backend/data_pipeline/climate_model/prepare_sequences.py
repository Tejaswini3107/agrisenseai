import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler


def prepare_sequences(csv_path, sequence_length=7, forecast_horizon=3):
    """
    Prepare sequential data for LSTM weather forecasting.

    Args:
        csv_path: Path to climate_dataset.csv
        sequence_length: Number of past days to use (default 7)
        forecast_horizon: Number of days to predict ahead (default 3)

    Returns:
        X_sequences: Input sequences (batch, sequence_length, 4)
        y_sequences: Target sequences (batch, forecast_horizon, 4)
        scalers: Dict of fitted scalers for each weather feature
        dates: Array of dates for the sequences
    """
    print("=" * 70)
    print("Preparing Sequential Weather Data for LSTM")
    print("=" * 70)

    # Load data
    print(f"\n1. Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows")

    # Sort by location and date to maintain temporal order
    print(f"\n2. Sorting data by location and date...")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    print(f"✓ Data sorted")

    # Select weather features to forecast
    weather_features = ["temperature", "humidity", "rainfall_mm", "wind_speed"]

    print(f"\n3. Preparing sequences (length={sequence_length}, horizon={forecast_horizon})...")

    X_sequences = []
    y_sequences = []
    sequence_dates = []

    # Normalize each feature
    scalers = {}
    for feature in weather_features:
        scalers[feature] = MinMaxScaler(feature_range=(0, 1))
        df[f"{feature}_scaled"] = scalers[feature].fit_transform(df[[feature]])

    scaled_features = [f"{f}_scaled" for f in weather_features]

    # Process each location separately to avoid data leakage
    for location in df["location"].unique():
        location_data = df[df["location"] == location].reset_index(drop=True)
        location_values = location_data[scaled_features].values
        location_dates = location_data["date"].values

        print(f"\n   Processing {location}...")

        # Create sequences
        for i in range(len(location_values) - sequence_length - forecast_horizon + 1):
            # Input: past sequence_length days
            X_seq = location_values[i : i + sequence_length]

            # Output: next forecast_horizon days
            y_seq = location_values[i + sequence_length : i + sequence_length + forecast_horizon]

            # Only add if we have complete forecast horizon
            if len(y_seq) == forecast_horizon:
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
                sequence_dates.append(location_dates[i + sequence_length - 1])

        print(f"   ✓ Created {len([x for x in X_sequences if x is not None])} sequences")

    # Convert to numpy arrays
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    sequence_dates = np.array(sequence_dates)

    print(f"\n4. Final sequences:")
    print(f"   - X shape: {X_sequences.shape} (samples, days, features)")
    print(f"   - y shape: {y_sequences.shape} (samples, forecast_days, features)")
    print(f"   - Total sequences: {len(X_sequences)}")

    # Create sequences directory
    os.makedirs("data/sequences", exist_ok=True)

    # Save sequences
    print(f"\n5. Saving sequences...")
    np.save("data/sequences/X_sequences.npy", X_sequences)
    np.save("data/sequences/y_sequences.npy", y_sequences)
    np.save("data/sequences/sequence_dates.npy", sequence_dates)
    joblib.dump(scalers, "data/sequences/scalers.pkl")
    joblib.dump(weather_features, "data/sequences/weather_features.pkl")

    print(f"✓ X_sequences.npy saved")
    print(f"✓ y_sequences.npy saved")
    print(f"✓ sequence_dates.npy saved")
    print(f"✓ scalers.pkl saved")
    print(f"✓ weather_features.pkl saved")

    print("\n" + "=" * 70)
    print("Sequence preparation completed successfully!")
    print("=" * 70)

    return X_sequences, y_sequences, scalers, sequence_dates


if __name__ == "__main__":
    X, y, scalers, dates = prepare_sequences("data/climate_dataset.csv", sequence_length=7, forecast_horizon=3)
