import os
import json

def save_forecast_json(forecast_df, output_file="outputs/forecast.json", last_n=None):
    """
    Saves forecast DataFrame as JSON and detects peak hours (top 5%).

    Parameters:
    - forecast_df: DataFrame with columns ['ds', 'yhat']
    - output_file: path to save JSON
    - last_n: if set, only save last N rows
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Ensure ds column exists and is string for JSON
    if "ds" not in forecast_df.columns or "yhat" not in forecast_df.columns:
        raise ValueError("❌ forecast_df must contain 'ds' and 'yhat' columns")

    if last_n is not None:
        forecast_df = forecast_df.tail(last_n)

    # Detect top 5% peaks
    threshold = forecast_df['yhat'].quantile(0.95)
    peaks = forecast_df[forecast_df['yhat'] >= threshold]['ds'].astype(str).tolist()

    # Build JSON output
    output = {
        "forecast": forecast_df.assign(ds=forecast_df['ds'].astype(str)).to_dict(orient="records"),
        "peaks": peaks
    }

    # Save
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Forecast JSON saved at {output_file}")
