import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

# --- Initialize session state ---
if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame()

from twilio.rest import Client

def send_alert(message):
    account_sid = os.getenv("TWILIO_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_NUMBER")
    to_number = os.getenv("TO_NUMBER")

    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_=twilio_number,
        to=to_number
    )

# --- Simulate sensor data ---
def generate_sensor_data(n=20):

    timestamps = pd.date_range(datetime.now(), periods=n, freq="s")
    data = {
        "Timestamp": timestamps,
        "pH": np.round(np.random.normal(7, 0.5, n), 2),
        "Turbidity": np.round(np.random.uniform(0, 5, n), 2),
        "TDS": np.round(np.random.uniform(100, 500, n), 2),
        "Temperature": np.round(np.random.uniform(20, 35, n), 2),
        "Nitrate": np.round(np.random.uniform(0, 50, n), 2),
        "Chloride": np.round(np.random.uniform(20, 250, n), 2),
        "Fluoride": np.round(np.random.uniform(0, 2, n), 2),
        "Hardness": np.round(np.random.uniform(80, 400, n), 2),
    }
    return pd.DataFrame(data)

# --- Custom threshold logic ---
def check_thresholds(df, thresholds):
    breach_flags = []
    for _, row in df.iterrows():
        breached = []
        for param, (low, high) in thresholds.items():
            if not (low <= row[param] <= high):
                breached.append(param)
        breach_flags.append(", ".join(breached) if breached else "None")
    df["Threshold_Breach"] = breach_flags
    return df

# --- UI Start ---
st.title("💧 Water Quality Monitoring Dashboard")

# Sidebar for data simulation
st.sidebar.header("Controls")
if st.sidebar.button("Simulate Data"):
    st.session_state.live_data = generate_sensor_data()

# Setup default thresholds
default_thresholds = {
    "pH": (6.5, 8.5),
    "Turbidity": (0, 5),
    "TDS": (0, 500),
    "Temperature": (5, 45),
    "Nitrate": (0, 45),
    "Chloride": (0, 250),
    "Fluoride": (0, 1.5),
    "Hardness": (0, 300),
}

# Editable thresholds in sidebar
st.sidebar.header("Custom Thresholds")
user_thresholds = {}
for param, (low, high) in default_thresholds.items():
    user_low = st.sidebar.number_input(f"{param} Min", value=low)
    user_high = st.sidebar.number_input(f"{param} Max", value=high)
    user_thresholds[param] = (user_low, user_high)

# Load existing or default data
if "live_data" not in st.session_state:
    st.session_state.live_data = generate_sensor_data()

# Check thresholds
live_data = check_thresholds(st.session_state.live_data.copy(), user_thresholds)

# --- AI Anomaly Detection ---
from sklearn.ensemble import IsolationForest

# Prepare data for model (exclude timestamp and breach info)
# --- Clean and prepare features for AI model ---
columns_to_drop = [col for col in ["Timestamp", "Threshold_Breach", "Sensor_Mismatch", "AI_Anomaly"] if col in live_data.columns]
features = live_data.drop(columns=columns_to_drop)

# Convert all values to numeric (if any strings slipped in)
# --- Final clean and safe version of AI anomaly detection ---
columns_to_drop = ["Timestamp", "Threshold_Breach", "Sensor_Mismatch", "AI_Anomaly"]
features = live_data.drop(columns=[col for col in columns_to_drop if col in live_data.columns])

# Force everything to numeric
features = features.apply(pd.to_numeric, errors="coerce")

# Keep only numeric columns
features = features.select_dtypes(include=[np.number])

# Drop rows with NaNs
features = features.dropna()

# Fit Isolation Forest only if valid data exists
if features.empty or features.shape[0] < 2:
    st.warning("Insufficient or invalid data for anomaly detection.")
else:
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=0.1, random_state=42)
    live_data["AI_Anomaly"] = model.fit_predict(features)


# Convert model output to readable label
if "AI_Anomaly" in live_data.columns:
    live_data["AI_Anomaly"] = live_data["AI_Anomaly"].map({1: "Normal", -1: "Anomaly"})
else:
    live_data["AI_Anomaly"] = "Not evaluated"

# Show table
st.subheader("🔍 Sensor Readings")
st.dataframe(live_data)

# Show warnings
breaches = live_data[live_data["Threshold_Breach"] != "None"]
if not breaches.empty:
    st.warning("🚨 Threshold breach detected in the following records:")
    st.dataframe(breaches)
else:
    st.success("✅ All parameters within safe range")

# Show AI Anomalies
st.subheader("🧠 AI Anomaly Detection")
ai_anomalies = live_data[live_data["AI_Anomaly"] == "Anomaly"]
if not ai_anomalies.empty:
    st.error("🤖 AI detected anomalies in the following records:")
    st.dataframe(ai_anomalies)
else:
    st.info("🧠 No anomalies detected by the AI model.")

# --- Trigger Twilio Alerts ---

# If any threshold breaches
if not breaches.empty:
    # Keep it short and useful (1–2 lines max)
    alert_msg = f"🚨 Water Alert! {row['Timestamp']} - Issue in {', '.join(row['Sensor_Mismatch'].split(', ')[:2])}."
    send_alert(alert_msg)

# If any AI anomalies
elif not ai_anomalies.empty:
    alert_msg = f"🤖 AI Anomaly Detected:\nSuspicious reading:\n{ai_anomalies.iloc[-1].to_dict()}"
    send_alert(alert_msg)

# Graph selection
st.subheader("📈 Parameter Visualization")
# List only parameters that actually exist in the data and avoid redundant (_2) columns
plot_columns = [col for col in default_thresholds.keys() if col in live_data.columns]
param_to_plot = st.selectbox("Select parameter to plot:", plot_columns)
# Plotting
def plot_parameter(df, param):
    if "Timestamp" not in df.columns or param not in df.columns:
        st.warning(f"Cannot plot '{param}'. Column missing in data.")
        return
    
    fig, ax = plt.subplots()
    ax.plot(df["Timestamp"], df[param], marker='o', label=param)
    ax.set_xlabel("Time")
    ax.set_ylabel(param)
    ax.set_title(f"Live Plot of {param}")
    ax.legend()
    st.pyplot(fig)


plot_parameter(live_data, param_to_plot)

# Export button
if st.button("📤 Export to CSV"):
    filename = f"water_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    live_data.to_csv(filename, index=False)
    st.success(f"Data exported as {filename}")
