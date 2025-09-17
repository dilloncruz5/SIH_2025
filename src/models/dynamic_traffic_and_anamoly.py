import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv(r"C:\Users\dillo\Downloads\SIH_2025-main\SIH_2025-main\data\preprocessed_dataset.csv")

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")

# Features (exclude target)
features = [c for c in df.columns if c not in ['datetime', 'Power demand']]

X = df[features]
y = df['Power demand']

# Split data (train on past, test on recent)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

# Train model (ONLY ONCE)
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# Create results dataframe
df_results = pd.DataFrame({
    "datetime": df['datetime'].iloc[-len(y_test):],
    "actual": y_test.values,  # Convert to numpy array to avoid index issues
    "predicted": y_pred
})

# Dynamic Tariff Assignment
def assign_tariff(demand):
    if demand > 0.7:
        return "High Tariff ⚠"
    elif demand > 0.4:
        return "Normal Tariff"
    else:
        return "Low Tariff 💡"

df_results["Tariff"] = df_results["predicted"].apply(assign_tariff)

# Anomaly Detection
df_results["error"] = abs(df_results["actual"] - df_results["predicted"])
df_results["Anomaly"] = np.where(
    df_results["error"] > 0.2 * df_results["predicted"],
    "⚠ Anomaly",
    "Normal"
)

# First Plot - Basic with anomalies highlighted
plt.figure(figsize=(12,6))
plt.plot(df_results["datetime"], df_results["actual"], label="Actual Demand", color="blue")
plt.plot(df_results["datetime"], df_results["predicted"], label="Predicted Demand", color="orange")

# Highlight anomalies (FIXED)
anomaly_mask = df_results["Anomaly"] == "⚠ Anomaly"
anomaly_data = df_results[anomaly_mask]
if len(anomaly_data) > 0:
    plt.scatter(anomaly_data["datetime"], anomaly_data["actual"], color="red", label=f"Anomaly ({len(anomaly_data)})")

plt.title("Power Demand Forecast with Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Demand (normalized)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Second Plot - Enhanced with Tariff Bands
plt.figure(figsize=(14,8))

# Normalize BOTH actual and predicted for consistent comparison
y_actual_norm = (df_results["actual"] - df_results["actual"].min()) / (df_results["actual"].max() - df_results["actual"].min())
y_pred_norm = (df_results["predicted"] - df_results["predicted"].min()) / (df_results["predicted"].max() - df_results["predicted"].min())

# Define tariff thresholds based on predicted values
low_threshold = 0.3
high_threshold = 0.7

# Plot normalized data
plt.plot(df_results["datetime"], y_actual_norm, label="Actual Demand", color="blue", linewidth=1.5)
plt.plot(df_results["datetime"], y_pred_norm, label="Predicted Demand", color="orange", alpha=0.8, linewidth=1.5)

# Add anomaly markers 
if len(anomaly_data) > 0:
    # Get normalized values for anomalies
    anomaly_indices = anomaly_data.index - df_results.index[0]  # Relative indices
    anomaly_actual_norm = y_actual_norm.iloc[anomaly_indices]
    
    plt.scatter(anomaly_data["datetime"], anomaly_actual_norm, 
               color="red", label=f"Anomalies ({len(anomaly_data)})", 
               s=50, zorder=5, alpha=0.8)

# Shade tariff bands
plt.axhspan(0, low_threshold, facecolor='green', alpha=0.15, label="Low Tariff Zone")
plt.axhspan(low_threshold, high_threshold, facecolor='yellow', alpha=0.15, label="Normal Tariff Zone")
plt.axhspan(high_threshold, 1, facecolor='red', alpha=0.15, label="High Tariff Zone")

# Add threshold lines for clarity
plt.axhline(y=low_threshold, color='green', linestyle='--', alpha=0.5)
plt.axhline(y=high_threshold, color='red', linestyle='--', alpha=0.5)

# Labels and legend
plt.title("Power Demand Forecast with Dynamic Tariff Bands & Anomaly Detection", fontsize=14, fontweight='bold')
plt.xlabel("Time")
plt.ylabel("Demand (Normalized)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("DEMAND FORECASTING SUMMARY")
print("="*50)
print(f"Total predictions: {len(df_results)}")
print(f"Anomalies detected: {len(anomaly_data)} ({len(anomaly_data)/len(df_results)*100:.1f}%)")
print(f"Average error: {df_results['error'].mean():.4f}")
print(f"Max error: {df_results['error'].max():.4f}")

# Tariff distribution
tariff_counts = df_results['Tariff'].value_counts()
print(f"\nTariff Distribution:")
for tariff, count in tariff_counts.items():
    percentage = (count / len(df_results)) * 100
    print(f"  {tariff}: {count} periods ({percentage:.1f}%)")

# Save results
df_results.to_csv("forecast_results.csv", index=False)
print(f"\nResults saved to 'forecast_results.csv'")