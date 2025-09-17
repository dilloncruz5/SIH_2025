# load_forecast.py

import json
import pandas as pd
from datetime import datetime, timedelta

def load_pregenerated_forecast(file_path):
    """
    Loads a pre-generated forecast from a JSON file.

    The JSON is expected to have a 'forecast' key containing a list of
    dictionaries with 'ds' (timestamp) and 'yhat' (demand) fields.

    Args:
        file_path (str): The path to the forecast JSON file.

    Returns:
        pd.DataFrame: A DataFrame with the loaded forecast data.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        forecast_data = data.get('forecast', [])
        
        if not forecast_data:
            raise ValueError(f"No 'forecast' data found in {file_path}")
            
        df = pd.DataFrame(forecast_data)
        df.rename(columns={'ds': 'timestamp', 'yhat': 'demand_kw'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    except FileNotFoundError:
        print(f"❌ Error: Forecast file not found at {file_path}. Have you run train_model.py?")
        return None
    except Exception as e:
        print(f"❌ An error occurred while loading the forecast file: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    forecast_path = "outputs/forecast_prophet_5min.json"
    forecast_df = load_pregenerated_forecast(forecast_path)
    if forecast_df is not None:
        print("Successfully loaded forecast.")
        print(forecast_df.head())
