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
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import requests
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-good {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .parameter-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .sidebar-header {
        color: #1f77b4;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .alert-box {
        background: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff0000;
    }
    
    .info-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame()
if "historical_data" not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False

# --- WHO/EPA Water Quality Standards ---
WHO_STANDARDS = {
    "pH": {"min": 6.5, "max": 8.5, "unit": "", "description": "Acidity/Alkalinity"},
    "Turbidity": {"min": 0, "max": 5, "unit": "NTU", "description": "Water Clarity"},
    "TDS": {"min": 0, "max": 500, "unit": "mg/L", "description": "Total Dissolved Solids"},
    "Temperature": {"min": 5, "max": 30, "unit": "°C", "description": "Water Temperature"},
    "Nitrate": {"min": 0, "max": 45, "unit": "mg/L", "description": "Nitrate Concentration"},
    "Chloride": {"min": 0, "max": 250, "unit": "mg/L", "description": "Chloride Content"},
    "Fluoride": {"min": 0, "max": 1.5, "unit": "mg/L", "description": "Fluoride Level"},
    "Hardness": {"min": 0, "max": 300, "unit": "mg/L", "description": "Water Hardness"},
    "DO": {"min": 4, "max": 14, "unit": "mg/L", "description": "Dissolved Oxygen"},
    "BOD": {"min": 0, "max": 30, "unit": "mg/L", "description": "Biochemical Oxygen Demand"}
}

# --- Telegram Alert Function ---
def send_telegram_alert(message):
    try:
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if bot_token and chat_id:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            requests.post(url, data=payload, timeout=5)
    except Exception as e:
        st.error(f"Telegram alert failed: {e}")

# --- Enhanced sensor data generation ---
def generate_sensor_data(n=50, add_anomaly=False):
    """Generate realistic sensor data with optional anomalies"""
    base_time = datetime.now()
    timestamps = [base_time - timedelta(seconds=i*10) for i in range(n)]
    timestamps.reverse()
    
    # Generate realistic water quality data
    data = {
        "Timestamp": timestamps,
        "pH": np.round(np.random.normal(7.2, 0.3, n), 2),
        "Turbidity": np.round(np.random.lognormal(0.5, 0.3, n), 2),
        "TDS": np.round(np.random.normal(300, 50, n), 2),
        "Temperature": np.round(np.random.normal(25, 3, n), 2),
        "Nitrate": np.round(np.random.lognormal(2, 0.5, n), 2),
        "Chloride": np.round(np.random.normal(150, 30, n), 2),
        "Fluoride": np.round(np.random.normal(0.8, 0.2, n), 2),
        "Hardness": np.round(np.random.normal(200, 40, n), 2),
        "DO": np.round(np.random.normal(8, 1, n), 2),
        "BOD": np.round(np.random.lognormal(2, 0.4, n), 2),
    }
    
    # Add some anomalies if requested
    if add_anomaly:
        anomaly_indices = np.random.choice(n, size=max(1, n//10), replace=False)
        for idx in anomaly_indices:
            data["pH"][idx] = np.random.choice([4.5, 9.5])
            data["Turbidity"][idx] = np.random.uniform(8, 15)
            data["TDS"][idx] = np.random.uniform(600, 800)
    
    return pd.DataFrame(data)

# --- Water Quality Assessment ---
def assess_water_quality(df):
    """Assess overall water quality based on WHO standards"""
    assessment_scores = []
    
    for _, row in df.iterrows():
        score = 0
        total_params = 0
        
        for param, standards in WHO_STANDARDS.items():
            if param in row:
                value = row[param]
                if standards["min"] <= value <= standards["max"]:
                    score += 1
                total_params += 1
        
        quality_score = (score / total_params) * 100 if total_params > 0 else 0
        assessment_scores.append(quality_score)
    
    df["Quality_Score"] = np.round(assessment_scores, 1)
    
    # Categorize water quality
    def categorize_quality(score):
        if score >= 90: return "Excellent"
        elif score >= 75: return "Good"
        elif score >= 60: return "Fair"
        elif score >= 40: return "Poor"
        else: return "Unacceptable"
    
    df["Quality_Category"] = df["Quality_Score"].apply(categorize_quality)
    return df

# --- Enhanced threshold checking ---
def check_thresholds(df):
    breach_details = []
    breach_count = []
    
    for _, row in df.iterrows():
        breached = []
        breach_types = []
        
        for param, standards in WHO_STANDARDS.items():
            if param in row:
                value = row[param]
                if value < standards["min"]:
                    breached.append(f"{param} (Low: {value})")
                    breach_types.append("Low")
                elif value > standards["max"]:
                    breached.append(f"{param} (High: {value})")
                    breach_types.append("High")
        
        breach_details.append("; ".join(breached) if breached else "None")
        breach_count.append(len(breached))
    
    df["Threshold_Breach"] = breach_details
    df["Breach_Count"] = breach_count
    return df

# --- AI Anomaly Detection ---
def detect_anomalies(df):
    """Enhanced anomaly detection with multiple algorithms"""
    try:
        # Select numeric columns for analysis
        numeric_cols = [col for col in WHO_STANDARDS.keys() if col in df.columns]
        features = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
        
        if features.empty or len(features) < 10:
            df["AI_Anomaly"] = "Insufficient Data"
            df["Anomaly_Score"] = 0
            return df
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features)
        anomaly_scores = iso_forest.score_samples(features)
        
        # Map results back to original dataframe
        df = df.copy()
        df["AI_Anomaly"] = "Normal"
        df["Anomaly_Score"] = 0
        
        for i, (idx, _) in enumerate(features.iterrows()):
            df.loc[idx, "AI_Anomaly"] = "Anomaly" if anomaly_labels[i] == -1 else "Normal"
            df.loc[idx, "Anomaly_Score"] = abs(anomaly_scores[i])
        
        return df
    
    except Exception as e:
        st.error(f"Anomaly detection failed: {e}")
        df["AI_Anomaly"] = "Error"
        df["Anomaly_Score"] = 0
        return df

# --- Main App ---
st.markdown('<h1 class="main-header">💧 Advanced Water Quality Monitoring System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time monitoring and analysis for water and wastewater engineering</p>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.markdown('<div class="sidebar-header">🎛️ System Controls</div>', unsafe_allow_html=True)

# Data generation controls
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Generate Data", type="primary"):
        with st.spinner("Generating sensor data..."):
            st.session_state.live_data = generate_sensor_data(n=100)
            st.success("Data generated!")

with col2:
    if st.button("⚠️ Add Anomalies"):
        with st.spinner("Adding anomalies..."):
            st.session_state.live_data = generate_sensor_data(n=100, add_anomaly=True)
            st.warning("Anomalous data generated!")

# Monitoring controls
st.sidebar.markdown("### 📡 Live Monitoring")
monitoring_interval = st.sidebar.slider("Update Interval (seconds)", 5, 60, 10)

if st.sidebar.button("▶️ Start Monitoring"):
    st.session_state.monitoring_active = True
    st.sidebar.success("Monitoring started!")

if st.sidebar.button("⏹️ Stop Monitoring"):
    st.session_state.monitoring_active = False
    st.sidebar.info("Monitoring stopped!")

# Custom thresholds
st.sidebar.markdown("### ⚙️ Custom Thresholds")
with st.sidebar.expander("Adjust Parameters"):
    custom_thresholds = {}
    for param, standards in WHO_STANDARDS.items():
        col1, col2 = st.columns(2)
        with col1:
            min_val = st.number_input(f"{param} Min", value=standards["min"], key=f"{param}_min")
        with col2:
            max_val = st.number_input(f"{param} Max", value=standards["max"], key=f"{param}_max")
        custom_thresholds[param] = {"min": min_val, "max": max_val}

# Initialize data if empty
if st.session_state.live_data.empty:
    st.session_state.live_data = generate_sensor_data()

# Process data
current_data = st.session_state.live_data.copy()
current_data = assess_water_quality(current_data)
current_data = check_thresholds(current_data)
current_data = detect_anomalies(current_data)

# --- Dashboard Metrics ---
if not current_data.empty:
    latest_reading = current_data.iloc[-1]
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🌡️ Temperature",
            value=f"{latest_reading['Temperature']}°C",
            delta=f"{latest_reading['Temperature'] - 25:.1f}°C from ideal"
        )
    
    with col2:
        st.metric(
            label="⚗️ pH Level",
            value=f"{latest_reading['pH']}",
            delta=f"{latest_reading['pH'] - 7:.1f} from neutral"
        )
    
    with col3:
        st.metric(
            label="💧 TDS",
            value=f"{latest_reading['TDS']} mg/L",
            delta=f"{latest_reading['TDS'] - 300:.0f} mg/L"
        )
    
    with col4:
        st.metric(
            label="🌊 Turbidity",
            value=f"{latest_reading['Turbidity']} NTU",
            delta=f"{latest_reading['Turbidity'] - 1:.1f} NTU"
        )
    
    with col5:
        quality_score = latest_reading['Quality_Score']
        st.metric(
            label="✅ Quality Score",
            value=f"{quality_score}%",
            delta=f"{latest_reading['Quality_Category']}"
        )

# --- Status Overview ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    # Overall Status
    breaches = current_data[current_data["Breach_Count"] > 0]
    anomalies = current_data[current_data["AI_Anomaly"] == "Anomaly"]
    
    if breaches.empty and anomalies.empty:
        st.markdown('<div class="status-good">🟢 SYSTEM STATUS: ALL CLEAR</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">🟡 SYSTEM STATUS: ATTENTION REQUIRED</div>', unsafe_allow_html=True)

with col2:
    # Quick stats
    st.info(f"""
    📊 **Quick Statistics**
    - Total Readings: {len(current_data)}
    - Threshold Breaches: {len(breaches)}
    - AI Anomalies: {len(anomalies)}
    - Avg Quality Score: {current_data['Quality_Score'].mean():.1f}%
    """)

# --- Interactive Visualizations ---
st.markdown("## 📈 Real-time Data Visualization")

# Parameter selection for detailed view
selected_params = st.multiselect(
    "Select parameters to visualize:",
    options=list(WHO_STANDARDS.keys()),
    default=["pH", "Temperature", "TDS", "Turbidity"]
)

if selected_params:
    # Create subplots
    fig = make_subplots(
        rows=len(selected_params), cols=1,
        subplot_titles=[f"{param} ({WHO_STANDARDS[param]['unit']})" for param in selected_params],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, param in enumerate(selected_params):
        if param in current_data.columns:
            # Add data line
            fig.add_trace(
                go.Scatter(
                    x=current_data["Timestamp"],
                    y=current_data[param],
                    mode='lines+markers',
                    name=param,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ),
                row=i+1, col=1
            )
            
            # Add threshold lines
            standards = WHO_STANDARDS[param]
            fig.add_hline(
                y=standards["max"], line_dash="dash", line_color="red",
                annotation_text=f"Max: {standards['max']}", row=i+1, col=1
            )
            fig.add_hline(
                y=standards["min"], line_dash="dash", line_color="orange",
                annotation_text=f"Min: {standards['min']}", row=i+1, col=1
            )
    
    fig.update_layout(
        height=200 * len(selected_params),
        title_text="Real-time Parameter Monitoring",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- Data Tables ---
st.markdown("## 📋 Detailed Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Readings", "🚨 Alerts", "🧠 AI Analysis", "📈 Quality Trends"])

with tab1:
    st.subheader("Latest Sensor Readings")
    
    # Color-code the dataframe based on quality
    def highlight_quality(row):
        if row['Quality_Category'] == 'Excellent':
            return ['background-color: #d4edda'] * len(row)
        elif row['Quality_Category'] == 'Good':
            return ['background-color: #fff3cd'] * len(row)
        elif row['Quality_Category'] in ['Fair', 'Poor']:
            return ['background-color: #f8d7da'] * len(row)
        else:
            return ['background-color: #f5c6cb'] * len(row)
    
    # Display formatted data
    display_data = current_data.tail(20).copy()
    display_data['Timestamp'] = display_data['Timestamp'].dt.strftime('%H:%M:%S')
    
    st.dataframe(
        display_data.style.apply(highlight_quality, axis=1),
        use_container_width=True,
        height=400
    )

with tab2:
    st.subheader("🚨 Active Alerts")
    
    # Threshold breaches
    if not breaches.empty:
        st.error(f"⚠️ {len(breaches)} Threshold Breach(es) Detected")
        
        breach_summary = breaches.groupby('Quality_Category').size().reset_index()
        breach_summary.columns = ['Quality Category', 'Count']
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(breach_summary)
        with col2:
            fig = px.pie(breach_summary, values='Count', names='Quality Category', 
                        title="Breach Distribution by Quality Category")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breach information
        st.subheader("Detailed Breach Information")
        breach_display = breaches[['Timestamp', 'Threshold_Breach', 'Quality_Score', 'Quality_Category']].copy()
        breach_display['Timestamp'] = breach_display['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(breach_display, use_container_width=True)
        
        # Send alerts
        if len(breaches) > 0:
            latest_breach = breaches.iloc[-1]
            alert_msg = f"""
🚨 *WATER QUALITY ALERT*
📅 Time: {latest_breach['Timestamp']}
⚠️ Issues: {latest_breach['Threshold_Breach']}
📊 Quality Score: {latest_breach['Quality_Score']}%
🏷️ Category: {latest_breach['Quality_Category']}
            """
            send_telegram_alert(alert_msg)
    else:
        st.success("✅ No threshold breaches detected")

with tab3:
    st.subheader("🧠 AI Anomaly Detection Results")
    
    if not anomalies.empty:
        st.warning(f"🤖 {len(anomalies)} AI Anomal(ies) Detected")
        
        # Anomaly distribution
        anomaly_scores = current_data['Anomaly_Score'].values
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=anomaly_scores, nbinsx=20, name="Anomaly Scores"))
        fig.update_layout(title="Distribution of Anomaly Scores", xaxis_title="Anomaly Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed anomaly information
        anomaly_display = anomalies[['Timestamp', 'AI_Anomaly', 'Anomaly_Score', 'Quality_Score']].copy()
        anomaly_display['Timestamp'] = anomaly_display['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(anomaly_display, use_container_width=True)
    else:
        st.success("✅ No anomalies detected by AI system")

with tab4:
    st.subheader("📈 Water Quality Trends")
    
    # Quality score over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=current_data["Timestamp"],
        y=current_data["Quality_Score"],
        mode='lines+markers',
        name='Quality Score',
        line=dict(color='blue', width=3),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="Water Quality Score Over Time",
        xaxis_title="Time",
        yaxis_title="Quality Score (%)",
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality distribution
    quality_dist = current_data['Quality_Category'].value_counts()
    fig = px.bar(x=quality_dist.index, y=quality_dist.values, 
                 title="Distribution of Water Quality Categories",
                 color=quality_dist.values,
                 color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)

# --- Export and Reports ---
st.markdown("---")
st.markdown("## 📤 Export & Reports")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Generate Report", type="primary"):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"water_quality_report_{timestamp}.csv"
        current_data.to_csv(filename, index=False)
        st.success(f"✅ Report saved as {filename}")

with col2:
    if st.button("📧 Email Report"):
        st.info("📧 Email functionality would be implemented here")

with col3:
    if st.button("📱 Send SMS Alert"):
        st.info("📱 SMS alert functionality would be implemented here")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🎓 <strong>Water & Wastewater Engineering Project</strong></p>
    <p>Built with Streamlit • Real-time monitoring • AI-powered analysis</p>
    <p>© 2025 - Advanced Water Quality Monitoring System</p>
</div>
""", unsafe_allow_html=True)

# --- Auto-refresh for live monitoring ---
if st.session_state.monitoring_active:
    time.sleep(monitoring_interval)
   # st.rerun()
# --- Export to CSV ---
if st.button("📤 Export to CSV"):
    filename = f"water_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    live_data.to_csv(filename, index=False)
    st.success(f"Data exported as {filename}")
