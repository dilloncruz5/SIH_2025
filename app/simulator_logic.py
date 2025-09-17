# simulator_logic.py

import pandas as pd
import numpy as np

def apply_dr_scenario(baseline_df, scenario):
    """
    Applies a demand response scenario to the baseline forecast.
    
    Args:
        baseline_df (pd.DataFrame): The original baseline demand forecast.
        scenario (dict): A dictionary defining the DR scenario.
            e.g., {'type': 'peak_reduction', 'start_hour': 17, 'end_hour': 20, 'reduction_percent': 15}
            e.g., {'type': 'ev_shift', 'shift_hours': 4, 'magnitude_kw': 20}
            
    Returns:
        pd.DataFrame: A new DataFrame with the adjusted demand forecast.
    """
    adjusted_df = baseline_df.copy()
    
    scenario_type = scenario.get('type')
    
    if scenario_type == 'peak_reduction':
        start_time = scenario.get('start_hour', 0)
        end_time = scenario.get('end_hour', 24)
        reduction = scenario.get('reduction_percent', 0) / 100.0
        
        mask = (adjusted_df['timestamp'].dt.hour >= start_time) & (adjusted_df['timestamp'].dt.hour < end_time)
        adjusted_df.loc[mask, 'demand_kw'] *= (1 - reduction)
        
    elif scenario_type == 'ev_shift':
        shift_hours = scenario.get('shift_hours', 0)
        magnitude_kw = scenario.get('magnitude_kw', 0)
        
        # Assume a specific charging window to shift, e.g., from 17:00-21:00
        original_charging_start_hour = 17
        original_charging_end_hour = 21
        
        # Determine the shifted window
        shifted_charging_start_hour = original_charging_start_hour + shift_hours
        shifted_charging_end_hour = original_charging_end_hour + shift_hours
        
        # Create a temporary load profile for the original charging
        original_load_profile = np.zeros(len(adjusted_df))
        original_load_mask = (adjusted_df['timestamp'].dt.hour >= original_charging_start_hour) & \
                             (adjusted_df['timestamp'].dt.hour < original_charging_end_hour)
        original_load_profile[original_load_mask] = magnitude_kw
        
        # Create a temporary load profile for the shifted charging
        shifted_load_profile = np.zeros(len(adjusted_df))
        shifted_load_mask = (adjusted_df['timestamp'].dt.hour >= shifted_charging_start_hour) & \
                            (adjusted_df['timestamp'].dt.hour < shifted_charging_end_hour)
        shifted_load_profile[shifted_load_mask] = magnitude_kw
        
        # Apply the shift to the adjusted forecast
        adjusted_df['demand_kw'] -= original_load_profile
        adjusted_df['demand_kw'] += shifted_load_profile
    
    return adjusted_df

def calculate_kpis(baseline_df, adjusted_df):
    """
    Calculates key performance indicators for the DR scenario.
    
    Args:
        baseline_df (pd.DataFrame): The original baseline forecast.
        adjusted_df (pd.DataFrame): The adjusted forecast.
        
    Returns:
        dict: A dictionary of calculated KPIs.
    """
    baseline_peak = baseline_df['demand_kw'].max()
    adjusted_peak = adjusted_df['demand_kw'].max()
    
    peak_reduction_kw = baseline_peak - adjusted_peak
    peak_reduction_pct = (peak_reduction_kw / baseline_peak) * 100 if baseline_peak != 0 else 0
    
    # Calculate the total energy shifted by summing the absolute difference
    total_energy_shifted = (adjusted_df['demand_kw'] - baseline_df['demand_kw']).abs().sum() / 12  # Divide by 12 for 5-min intervals
    
    return {
        'baseline_peak': baseline_peak,
        'adjusted_peak': adjusted_peak,
        'peak_reduction_kw': peak_reduction_kw,
        'peak_reduction_pct': peak_reduction_pct,
        'total_energy_shifted': total_energy_shifted
    }
