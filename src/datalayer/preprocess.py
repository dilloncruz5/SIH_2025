import pandas as pd
import numpy as np
import os

def preprocess_5min(demand_file, weather_file=None, save_path="data/processed_delhi_demand.csv"):
    # ----------------------------
    # 1. Load 5-minute demand data
    # ----------------------------
    demand = pd.read_csv(demand_file, parse_dates=["datetime"])
    
    # ----------------------------
    # 2. Resample to hourly demand
    # ----------------------------
    demand = demand.set_index("datetime")
    hourly_demand = demand.resample("H").mean().reset_index()
    
    # ----------------------------
    # 3. Merge weather data if provided
    # ----------------------------
    if weather_file is not None:
        weather = pd.read_csv(weather_file, parse_dates=["datetime"])
        data = pd.merge(hourly_demand, weather, on="datetime", how="left")
        # Fill missing weather values
        data.fillna(method='ffill', inplace=True)
    else:
        data = hourly_demand
    
    # ----------------------------
    # 4. Feature Engineering
    # ----------------------------
    data["hour"] = data["datetime"].dt.hour
    data["day_of_week"] = data["datetime"].dt.dayofweek
    data["month"] = data["datetime"].dt.month
    data["weekend"] = data["day_of_week"].isin([5,6]).astype(int)
    
    # Lag features
    data["lag_1h"] = data["demand"].shift(1)
    data["lag_24h"] = data["demand"].shift(24)
    
    # Rolling averages
    data["rolling_3h"] = data["demand"].rolling(window=3).mean()
    data["rolling_7d"] = data["demand"].rolling(window=24*7).mean()
    
    # ----------------------------
    # 5. Handle missing values
    # ----------------------------
    data.dropna(inplace=True)
    
    # ----------------------------
    # 6. Save processed dataset
    # ----------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path, index=False)
    
    print(f"✅ Cleaned and processed dataset saved at {save_path}")
    return data

# ----------------------------
# If run as script
# ----------------------------
if __name__ == "__main__":
    preprocess_5min("data/raw_delhi_demand.csv", "data/raw_weather.csv")