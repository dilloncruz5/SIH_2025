import os
import json

def save_forecast_json(forecast_df, output_file="outputs/forecast.json", last_n=None):
    """
    Saves forecast DataFrame as JSON and detects peak hours (top 5%).
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Use only last_n rows if specified
    if last_n is not None:
        forecast_df = forecast_df.tail(last_n)

    # Determine top 5% peaks
    threshold = forecast_df['yhat'].quantile(0.95)
    peaks = forecast_df[forecast_df['yhat'] >= threshold][['ds', 'yhat']]

    # Convert Timestamps to string for JSON serialization
    output = {
        "forecast": [
            {"ds": str(row["ds"]), "yhat": row["yhat"]} 
            for _, row in forecast_df.iterrows()
        ],
        "peaks": [str(ts) for ts in peaks['ds'].tolist()]
    }

    # Save JSON
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Forecast JSON saved at {output_file}")