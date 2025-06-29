import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# --- Simulate sensor data ---
def generate_sensor_data(n=20):

    # --- Twilio Alert Function ---
from twilio.rest import Client

def send_alert(message):
    # Your Twilio credentials
    account_sid = "TWILIO_SID"
    auth_token = "TWILIO_AUTH"
    twilio_number = "TWILIO_FROM"  # E.g., "+1234567890"

    # Destination number (your personal number)
    to_number = "ALERT_NUMBER"  # or "sms:+91xxxxxxxxxx"

    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_=twilio_number,
        to=to_number
    )

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
features = live_data.drop(columns=["Timestamp", "Threshold_Breach"])
model = IsolationForest(contamination=0.1, random_state=42)
live_data["AI_Anomaly"] = model.fit_predict(features)

# Convert model output to readable label
live_data["AI_Anomaly"] = live_data["AI_Anomaly"].map({1: "Normal", -1: "Anomaly"})

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

# Graph selection
st.subheader("📈 Parameter Visualization")
param_to_plot = st.selectbox("Select parameter to plot:", list(default_thresholds.keys()))

# Plotting
def plot_parameter(df, param):
    fig, ax = plt.subplots()
    ax.plot(df["Timestamp"], df[param], marker='o', label=param)
    ax.axhline(user_thresholds[param][0], color='red', linestyle='--', label="Min Threshold")
    ax.axhline(user_thresholds[param][1], color='red', linestyle='--', label="Max Threshold")
    ax.set_xlabel("Time")
    ax.set_ylabel(param)
    ax.set_title(f"{param} Over Time")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

plot_parameter(live_data, param_to_plot)

# Export button
if st.button("📤 Export to CSV"):
    filename = f"water_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    live_data.to_csv(filename, index=False)
    st.success(f"Data exported as {filename}")
