from prophet_model import train_prophet
from lstm_model import train_lstm

if __name__ == "__main__":
    # ----------------------------
    # Prophet training
    # ----------------------------
    print("⏳ Starting Prophet training...")
    prophet_model, prophet_5min, prophet_hourly, prophet_json_5min, prophet_json_hourly = train_prophet(
        processed_file="data/preprocessed_dataset.csv",
        output_json_5min="outputs/forecast_prophet_5min.json",
        output_json_hourly="outputs/forecast_prophet_hourly.json"
    )
    print("✅ Prophet training completed.")
    print(f"   📂 5-min forecast saved at: {prophet_json_5min}")
    print(f"   📂 Hourly forecast saved at: {prophet_json_hourly}\n")

    # ----------------------------
    # LSTM training
    # ----------------------------
    print("⏳ Starting LSTM training...")
    lstm_model, lstm_5min, lstm_hourly = train_lstm(
        processed_file="data/preprocessed_dataset.csv",
        output_json_5min="outputs/forecast_lstm_5min.json",
        output_json_hourly="outputs/forecast_lstm_hourly.json"
    )
    print("✅ LSTM training completed.")
    print("   📂 5-min forecast saved at: outputs/forecast_lstm_5min.json")
    print("   📂 Hourly forecast saved at: outputs/forecast_lstm_hourly.json\n")

    print("🎯 All models trained and forecasts generated successfully!")
