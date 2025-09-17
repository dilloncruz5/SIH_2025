import pandas as pd
from prophet import Prophet
import joblib
import os
from utils import save_forecast_json

def train_prophet(
    processed_file="data/preprocessed_dataset.csv", 
    model_file="saved_models/prophet_model.pkl", 
    output_json_5min="outputs/forecast_prophet_5min.json",
    output_json_hourly="outputs/forecast_prophet_hourly.json",
    verbose=False  # new argument
):
    # ----------------------------
    # 1. Load processed dataset
    # ----------------------------
    data = pd.read_csv(processed_file, parse_dates=["datetime"], dayfirst=True)
    
    # ----------------------------
    # 2. Prepare data for Prophet
    # ----------------------------
    df = data[["datetime", "Power demand"]].rename(columns={"datetime": "ds", "Power demand": "y"})
    
    # ----------------------------
    # 3. Initialize and train Prophet
    # ----------------------------
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(df)  # CmdStan progress shown if verbose=True
    
   # ----------------------------
    # 4. Forecast next 24 hours at 5-min intervals
    # ----------------------------
    periods_5min = 24*12  # 24 hours × 12 = 288 steps of 5-min each
    future_5min = model.make_future_dataframe(periods=periods_5min, freq='5min')  # changed from '5T' to '5min'

    print("⏳ Generating 5-min forecast...")
    forecast_5min = model.predict(future_5min)
    print("✅ 5-min forecast generated. Resampling to hourly...")

    forecast_5min.set_index('ds', inplace=True)
    forecast_hourly = forecast_5min['yhat'].resample('H').mean().reset_index()
    print("✅ Hourly forecast generated. Saving JSONs...")

    save_forecast_json(forecast_5min, output_json_5min, last_n=288)
    save_forecast_json(forecast_hourly, output_json_hourly, last_n=24)

    print("✅ Forecasts saved. Prophet training complete!")
    
    # ----------------------------
    # 5. Save trained model
    # ----------------------------
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    joblib.dump(model, model_file)
    print(f"✅ Prophet model saved at {model_file}")
    
    # ----------------------------
    # 6. Save 5-min forecast JSON
    # ----------------------------
    save_forecast_json(forecast_5min, output_json_5min, last_n=288)
    
    # ----------------------------
    # 7. Aggregate hourly forecast for visualization
    # ----------------------------
    forecast_5min.set_index('ds', inplace=True)
    forecast_hourly = forecast_5min['yhat'].resample('H').mean().reset_index()
    save_forecast_json(forecast_hourly, output_json_hourly, last_n=24)
    
    return model, forecast_5min, forecast_hourly, output_json_5min, output_json_hourly


# ----------------------------
# If run as script
# ----------------------------
if __name__ == "__main__":
    prophet_model, prophet_5min, prophet_hourly = train_prophet(
        processed_file="data/preprocessed_dataset.csv",
        output_json_5min="outputs/forecast_prophet_5min.json",
        output_json_hourly="outputs/forecast_prophet_hourly.json",
        verbose=True  # now you can see CmdStan progress
    )