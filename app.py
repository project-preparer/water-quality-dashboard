import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import requests

# --- Initialize session state ---
if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame()

# --- Telegram Alert Function ---
def send_telegram_alert(message):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Telegram alert failed: {e}")

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

columns_to_drop = ["Timestamp", "Threshold_Breach", "Sensor_Mismatch", "AI_Anomaly"]
features = live_data.drop(columns=[col for col in columns_to_drop if col in live_data.columns])
features = features.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number]).dropna()

if features.empty or features.shape[0] < 2:
    st.warning("Insufficient or invalid data for anomaly detection.")
else:
    model = IsolationForest(contamination=0.1, random_state=42)
    live_data["AI_Anomaly"] = model.fit_predict(features)

if "AI_Anomaly" in live_data.columns:
    live_data["AI_Anomaly"] = live_data["AI_Anomaly"].map({1: "Normal", -1: "Anomaly"})
else:
    live_data["AI_Anomaly"] = "Not evaluated"

# Show table
st.subheader("🔍 Sensor Readings")
st.dataframe(live_data)

# Threshold Breach Display
breaches = live_data[live_data["Threshold_Breach"] != "None"]
if not breaches.empty:
    st.warning("🚨 Threshold breach detected in the following records:")
    st.dataframe(breaches)
else:
    st.success("✅ All parameters within safe range")

# AI Anomaly Display
st.subheader("🧠 AI Anomaly Detection")
ai_anomalies = live_data[live_data["AI_Anomaly"] == "Anomaly"]
if not ai_anomalies.empty:
    st.error("🤖 AI detected anomalies in the following records:")
    st.dataframe(ai_anomalies)
else:
    st.info("🧠 No anomalies detected by the AI model.")

# --- Telegram Alerts ---
if not breaches.empty:
    last = live_data.iloc[-1]
    breached = last['Threshold_Breach'].split(', ')[:2]
    alert_msg = (
        f"🚨 *Threshold Breach Alert!*\n\n"
        f"📅 Timestamp: {last['Timestamp']}\n"
        f"⚠️ Breached Parameters: {', '.join(breached)}\n"
        f"📊 Readings:\n"
        f"- pH: {last['pH']}\n"
        f"- Turbidity: {last['Turbidity']}\n"
        f"- TDS: {last['TDS']} ppm\n"
        f"- Temperature: {last['Temperature']} °C\n"
        f"- Nitrate: {last['Nitrate']} mg/L\n"
        f"- Chloride: {last['Chloride']} mg/L\n"
        f"- Fluoride: {last['Fluoride']} mg/L\n"
        f"- Hardness: {last['Hardness']} mg/L"
    )
    send_telegram_alert(alert_msg)


if not ai_anomalies.empty:
    last = ai_anomalies.iloc[-1]
    alert_msg = (
        f"🤖 *AI Anomaly Detected!*\n\n"
        f"📅 Timestamp: {last['Timestamp']}\n"
        f"📊 Parameters:\n"
        f"- pH: {last['pH']}\n"
        f"- Turbidity: {last['Turbidity']}\n"
        f"- TDS: {last['TDS']} ppm\n"
        f"- Temperature: {last['Temperature']} °C\n"
        f"- Nitrate: {last['Nitrate']} mg/L\n"
        f"- Chloride: {last['Chloride']} mg/L\n"
        f"- Fluoride: {last['Fluoride']} mg/L\n"
        f"- Hardness: {last['Hardness']} mg/L"
    )
    send_telegram_alert(alert_msg)

# --- Graph section ---
st.subheader("📈 Parameter Visualization")
plot_columns = [col for col in default_thresholds.keys() if col in live_data.columns]
param_to_plot = st.selectbox("Select parameter to plot:", plot_columns)

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

# --- Export to CSV ---
if st.button("📤 Export to CSV"):
    filename = f"water_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    live_data.to_csv(filename, index=False)
    st.success(f"Data exported as {filename}")
