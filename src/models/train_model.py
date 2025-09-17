from prophet_model import train_prophet
from lstm_model import train_lstm

if __name__ == "__main__":
    # ----------------------------
    # Prophet training
    # ----------------------------
    print("⏳ Starting Prophet training...")
    print("⏳ Training Prophet model...")
    prophet_model, prophet_5min, prophet_hourly, prophet_json_5min, prophet_json_hourly = train_prophet(
        processed_file="data/preprocessed_dataset.csv",
        output_json_5min="outputs/forecast_prophet_5min.json",
        output_json_hourly="outputs/forecast_prophet_hourly.json",
        verbose=True  # show CmdStan progress
    )
    print("✅ Prophet model trained.")
    
    print("⏳ Generating 5-min forecast...")
    # 5-min forecast is already generated inside train_prophet
    print("✅ 5-min forecast generated.")
    
    print("⏳ Aggregating to hourly forecast...")
    # hourly aggregation also done inside train_prophet
    print("✅ Hourly forecast generated.")
    
    print("⏳ Saving Prophet forecasts to JSONs...")
    print(f" - 5-min forecast: {prophet_json_5min}")
    print(f" - Hourly forecast: {prophet_json_hourly}")
    print("✅ Prophet forecasts saved.\n")
    
    # ----------------------------
    # LSTM training
    # ----------------------------
    print("⏳ Starting LSTM training...")
    lstm_model, lstm_5min, lstm_hourly = train_lstm(
        processed_file="data/preprocessed_dataset.csv",
        output_json_5min="outputs/forecast_lstm_5min.json",
        output_json_hourly="outputs/forecast_lstm_hourly.json"
    )
    print("✅ LSTM training completed!")