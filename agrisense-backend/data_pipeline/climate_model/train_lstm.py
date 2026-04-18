import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping

def train_lstm_weather_forecast():
    """
    Train LSTM model for weather forecasting.
    Predicts next 3 days of weather using 7 days of history.
    """
    print("=" * 70)
    print("Training LSTM Weather Forecasting Model")
    print("=" * 70)
    
    # Check if sequences are prepared
    print("\n1. Loading prepared sequences...")
    try:
        X = np.load('data/sequences/X_sequences.npy')
        y = np.load('data/sequences/y_sequences.npy')
        dates = np.load('data/sequences/sequence_dates.npy')
        scalers = joblib.load('data/sequences/scalers.pkl')
        weather_features = joblib.load('data/sequences/weather_features.pkl')
        
        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")
        print(f"✓ Weather features: {weather_features}")
    except FileNotFoundError:
        print("✗ Error: Sequence files not found!")
        print("  Please run: python -m data_pipeline.climate_model.prepare_sequences")
        return
    
    # Split into train/test (80/20)
    print("\n2. Splitting data (80% train, 20% test)...")
    split_idx = int(0.8 * len(X))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")
    print(f"✓ Input shape: {X_train.shape}")
    print(f"✓ Output shape: {y_train.shape}")
    
    # Build LSTM model
    print("\n3. Building LSTM model...")
    sequence_length = X_train.shape[1]  # 7 days
    num_features = X_train.shape[2]     # 4 features (temp, humidity, rainfall, wind)
    forecast_horizon = y_train.shape[1]  # 3 days
    
    model = Sequential([
        # LSTM layers with dropout for regularization
        layers.LSTM(64, activation='relu', return_sequences=True, 
                   input_shape=(sequence_length, num_features),
                   name='lstm_1'),
        layers.Dropout(0.2),
        
        layers.LSTM(32, activation='relu', return_sequences=False,
                   name='lstm_2'),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(32, activation='relu', name='dense_1'),
        layers.Dropout(0.2),
        
        # Output layer: predict all features for all future days
        layers.Dense(forecast_horizon * num_features, activation='linear'),
        layers.Reshape((forecast_horizon, num_features))
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    model.summary()
    print(f"✓ Model built successfully")
    
    # Train model
    print("\n4. Training LSTM model...")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("✓ Model training completed")
    
    # Evaluate on test set
    print("\n5. Evaluating model on test set...")
    
    y_pred_train = model.predict(X_train, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)
    
    # Inverse transform to original scale
    print("\n6. Inverse transforming predictions...")
    
    # Reshape for inverse transform
    y_train_original = np.zeros_like(y_train)
    y_pred_train_original = np.zeros_like(y_pred_train)
    y_test_original = np.zeros_like(y_test)
    y_pred_test_original = np.zeros_like(y_pred_test)
    
    for i, feature in enumerate(weather_features):
        scaler = scalers[feature]
        
        # Reshape for inverse transform
        y_train_reshaped = y_train[:, :, i].reshape(-1, 1)
        y_pred_train_reshaped = y_pred_train[:, :, i].reshape(-1, 1)
        y_test_reshaped = y_test[:, :, i].reshape(-1, 1)
        y_pred_test_reshaped = y_pred_test[:, :, i].reshape(-1, 1)
        
        # Inverse transform
        y_train_original[:, :, i] = scaler.inverse_transform(y_train_reshaped).reshape(y_train[:, :, i].shape)
        y_pred_train_original[:, :, i] = scaler.inverse_transform(y_pred_train_reshaped).reshape(y_pred_train[:, :, i].shape)
        y_test_original[:, :, i] = scaler.inverse_transform(y_test_reshaped).reshape(y_test[:, :, i].shape)
        y_pred_test_original[:, :, i] = scaler.inverse_transform(y_pred_test_reshaped).reshape(y_pred_test[:, :, i].shape)
    
    # Calculate metrics
    print("\n7. Calculating metrics...")
    
    train_mae = mean_absolute_error(y_train_original.reshape(-1), y_pred_train_original.reshape(-1))
    test_mae = mean_absolute_error(y_test_original.reshape(-1), y_pred_test_original.reshape(-1))
    
    train_rmse = np.sqrt(mean_squared_error(y_train_original.reshape(-1), y_pred_train_original.reshape(-1)))
    test_rmse = np.sqrt(mean_squared_error(y_test_original.reshape(-1), y_pred_test_original.reshape(-1)))
    
    print(f"\n✓ Training Metrics:")
    print(f"   - MAE: {train_mae:.4f}")
    print(f"   - RMSE: {train_rmse:.4f}")
    
    print(f"\n✓ Testing Metrics:")
    print(f"   - MAE: {test_mae:.4f}")
    print(f"   - RMSE: {test_rmse:.4f}")
    
    # Per-feature metrics
    print(f"\n✓ Per-Feature Test Metrics:")
    for i, feature in enumerate(weather_features):
        feature_mae = mean_absolute_error(y_test_original[:, :, i].reshape(-1), 
                                         y_pred_test_original[:, :, i].reshape(-1))
        feature_rmse = np.sqrt(mean_squared_error(y_test_original[:, :, i].reshape(-1), 
                                                  y_pred_test_original[:, :, i].reshape(-1)))
        print(f"   - {feature}: MAE={feature_mae:.4f}, RMSE={feature_rmse:.4f}")
    
    # Save model
    print("\n8. Saving model and artifacts...")
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/lstm_weather_model.keras'
    model.save(model_path)
    print(f"✓ LSTM model saved to {model_path}")
    
    # Save history
    history_path = 'models/lstm_training_history.pkl'
    joblib.dump(history.history, history_path)
    print(f"✓ Training history saved to {history_path}")
    
    # Generate plots
    print("\n9. Generating visualizations...")
    
    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('LSTM Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('LSTM MAE History')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_fig_path = 'models/lstm_training_history.png'
    plt.savefig(history_fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {history_fig_path}")
    plt.close()
    
    # Predictions vs Actual (for first test sample across 3 days)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(weather_features):
        axes[i].plot(y_test_original[0, :, i], 'o-', label='Actual', linewidth=2, markersize=8)
        axes[i].plot(y_pred_test_original[0, :, i], 's--', label='Predicted', linewidth=2, markersize=8)
        axes[i].set_xlabel('Days Ahead')
        axes[i].set_ylabel(feature.replace('_', ' ').title())
        axes[i].set_title(f'{feature.replace("_", " ").title()} Forecast (Sample 1)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    pred_fig_path = 'models/lstm_sample_predictions.png'
    plt.savefig(pred_fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Predictions plot saved to {pred_fig_path}")
    plt.close()
    
    # Final summary
    print("\n" + "=" * 70)
    print("LSTM Weather Forecasting Model Training Complete!")
    print("=" * 70)
    print(f"\n📊 Model Summary:")
    print(f"   - Architecture: LSTM(64) → LSTM(32) → Dense(32)")
    print(f"   - Input: 7 days of weather history")
    print(f"   - Output: 3-day weather forecast")
    print(f"   - Features: {', '.join(weather_features)}")
    print(f"   - Test MAE: {test_mae:.4f}")
    print(f"   - Test RMSE: {test_rmse:.4f}")
    print(f"\n📁 Saved Files:")
    print(f"   - {model_path}")
    print(f"   - {history_path}")
    print(f"   - {history_fig_path}")
    print(f"   - {pred_fig_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    train_lstm_weather_forecast()
