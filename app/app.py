# app.py

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Import the logic from the new file and the simulator logic
from load_forecast import load_pregenerated_forecast
from simulator_logic import apply_dr_scenario, calculate_kpis

# Set up the Streamlit page
st.set_page_config(page_title="Adaptive Demand Response Simulator", layout="wide")

# App title and description
st.title("⚡️ Adaptive Demand Response Simulator")
st.markdown("This simulator uses a pre-trained Prophet forecast. Use the sliders to simulate demand response scenarios.")

# --- Load the pre-trained forecast ---
forecast_path = "outputs/forecast_prophet_5min.json"
baseline_df = load_pregenerated_forecast(forecast_path)

if baseline_df is None:
    st.error("Forecast data could not be loaded. Please ensure you have run the `train_model.py` script to generate the forecast files.")
    st.stop() # Stops the app if the data isn't available

baseline_df.set_index('timestamp', inplace=True)

# --- Sidebar for user input ---
st.sidebar.header("Scenario Controls")
scenario_type = st.sidebar.selectbox("Select Scenario Type", ["Evening Peak Reduction", "EV Load Shifting"])

# Define sliders based on scenario type
if scenario_type == "Evening Peak Reduction":
    st.sidebar.subheader("Evening Peak Reduction")
    start_hour = st.sidebar.slider("Start Hour", min_value=17, max_value=20, value=18)
    end_hour = st.sidebar.slider("End Hour", min_value=18, max_value=23, value=21)
    reduction_pct = st.sidebar.slider("Reduction Percentage (%)", min_value=0, max_value=30, value=15, step=1)
    
    scenario = {
        'type': 'peak_reduction',
        'start_hour': start_hour,
        'end_hour': end_hour,
        'reduction_percent': reduction_pct
    }
    
elif scenario_type == "EV Load Shifting":
    st.sidebar.subheader("EV Load Shifting")
    shift_hours = st.sidebar.slider("Shift EV Charging (hours)", min_value=0, max_value=8, value=4)
    magnitude_kw = st.sidebar.slider("EV Charging Magnitude (kW)", min_value=0, max_value=50, value=25, step=5)
    
    scenario = {
        'type': 'ev_shift',
        'shift_hours': shift_hours,
        'magnitude_kw': magnitude_kw
    }

# --- Main app logic ---
adjusted_df = apply_dr_scenario(baseline_df.reset_index(), scenario)
adjusted_df.set_index('timestamp', inplace=True)

# Calculate KPIs
kpis = calculate_kpis(baseline_df, adjusted_df)

# --- Visualization ---
st.header("Load Profile Visualization")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=baseline_df.index,
    y=baseline_df['demand_kw'],
    mode='lines',
    name='Baseline Forecast',
    line=dict(color='#0077b6')
))

fig.add_trace(go.Scatter(
    x=adjusted_df.index,
    y=adjusted_df['demand_kw'],
    mode='lines',
    name='Adjusted Forecast',
    line=dict(color='#ef233c')
))

fig.update_layout(
    xaxis_title="Time of Day",
    yaxis_title="Demand (kW)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- KPI Section ---
st.header("Key Performance Indicators (KPIs)")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Baseline Peak Load", f"{kpis['baseline_peak']:.2f} kW")
    
with col2:
    st.metric("Adjusted Peak Load", f"{kpis['adjusted_peak']:.2f} kW")

with col3:
    st.metric("Peak Load Reduction", f"{kpis['peak_reduction_pct']:.2f}%", f"{kpis['peak_reduction_kw']:.2f} kW")
