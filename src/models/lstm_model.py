import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from utils import save_forecast_json
from tqdm.keras import TqdmCallback

def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm(processed_file="data/preprocessed_dataset.csv",
               model_file="saved_models/lstm_model.h5",
               scaler_file="saved_models/demand_scaler.pkl",
               output_json_5min="outputs/forecast_lstm_5min.json",
               output_json_hourly="outputs/forecast_lstm_hourly.json",
               seq_length=24, epochs=30, batch_size=16):
    
    # 1. Load dataset
    data = pd.read_csv(processed_file, parse_dates=["datetime"], dayfirst=True)
    demand = data["Power demand"].values.reshape(-1,1)
    
    # 2. Scale demand
    scaler = MinMaxScaler()
    demand_scaled = scaler.fit_transform(demand)
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(scaler, scaler_file)
    
    # 3. Create sequences
    X, y = create_sequences(demand_scaled, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # 4. Train/Test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 5. Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(seq_length,1)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # 6. Train model with progress
    print("⏳ Training LSTM model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[TqdmCallback(verbose=1)]
    )
    print("✅ LSTM training completed.")
    
    # 7. Save model
    model.save(model_file)
    print(f"✅ LSTM model saved at {model_file}")
    
    # 8. Forecast next 24h at 5-min intervals
    last_seq = demand_scaled[-seq_length:]
    input_seq = np.expand_dims(last_seq, axis=0)  # shape (1, seq_length, 1)
    forecast_scaled = []

    steps_5min = 24 * 12  # 288 steps
    print("⏳ Generating 5-min forecast...")
    for i in range(steps_5min):
        pred = model.predict(input_seq, verbose=0)
        forecast_scaled.append(pred[0,0])
        # reshape pred to (1,1,1) before appending
        input_seq = np.append(input_seq[:,1:,:], pred.reshape(1,1,1), axis=1)
        if (i+1) % 12 == 0:
            print(f"Progress: {(i+1)/steps_5min*100:.1f}%")
    print("✅ 5-min forecast generated.")
    
    forecast_scaled = np.array(forecast_scaled).reshape(-1,1)
    forecast = scaler.inverse_transform(forecast_scaled)
    
    forecast_df_5min = pd.DataFrame({
        "ds": pd.date_range(
            start=data["datetime"].iloc[-1] + pd.Timedelta(minutes=5),
            periods=steps_5min,
            freq="5min"
        ),
        "yhat": forecast.flatten()
    })
    
    # 9. Aggregate hourly
    forecast_df_hourly = forecast_df_5min.set_index("ds")["yhat"].resample("h").mean().reset_index()
    
    # 10. Save forecast JSON
    print("⏳ Saving forecast JSONs...")
    save_forecast_json(forecast_df_5min, output_json_5min, last_n=288)
    save_forecast_json(forecast_df_hourly, output_json_hourly, last_n=24)
    print("✅ Forecasts saved.")
    
    return model, forecast_df_5min, forecast_df_hourly

if __name__ == "__main__":
    train_lstm()
