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
    
    # Generate realistic water quality data with some natural variation
    data = {
        "Timestamp": timestamps,
        "pH": np.round(np.random.normal(7.2, 0.5, n), 2),  # Increased variation
        "Turbidity": np.round(np.random.lognormal(0.8, 0.6, n), 2),  # More variation
        "TDS": np.round(np.random.normal(350, 80, n), 2),  # Increased variation
        "Temperature": np.round(np.random.normal(25, 4, n), 2),  # More variation
        "Nitrate": np.round(np.random.lognormal(2.5, 0.8, n), 2),  # More variation
        "Chloride": np.round(np.random.normal(180, 50, n), 2),  # Increased variation
        "Fluoride": np.round(np.random.normal(1.0, 0.4, n), 2),  # More variation
        "Hardness": np.round(np.random.normal(220, 60, n), 2),  # Increased variation
        "DO": np.round(np.random.normal(7, 2, n), 2),  # More variation
        "BOD": np.round(np.random.lognormal(2.5, 0.6, n), 2),  # More variation
    }
    
    # Ensure some values are outside normal ranges for realistic breaches
    for i in range(n):
        if np.random.random() < 0.15:  # 15% chance of parameter being out of range
            param = np.random.choice(list(data.keys())[1:])  # Skip timestamp
            if param == "pH":
                data[param][i] = np.random.choice([np.random.uniform(4, 6), np.random.uniform(9, 10)])
            elif param == "Turbidity":
                data[param][i] = np.random.uniform(6, 12)
            elif param == "TDS":
                data[param][i] = np.random.uniform(550, 700)
            elif param == "Temperature":
                data[param][i] = np.random.choice([np.random.uniform(2, 4), np.random.uniform(35, 40)])
            elif param == "Nitrate":
                data[param][i] = np.random.uniform(50, 80)
            elif param == "Chloride":
                data[param][i] = np.random.uniform(280, 400)
            elif param == "Fluoride":
                data[param][i] = np.random.uniform(2, 3)
            elif param == "Hardness":
                data[param][i] = np.random.uniform(350, 500)
            elif param == "DO":
                data[param][i] = np.random.uniform(1, 3)
            elif param == "BOD":
                data[param][i] = np.random.uniform(35, 60)
    
    # Add more severe anomalies if requested
    if add_anomaly:
        anomaly_indices = np.random.choice(n, size=max(3, n//8), replace=False)
        for idx in anomaly_indices:
            # Create multiple parameter anomalies
            num_params = np.random.randint(2, 4)
            params_to_anomalize = np.random.choice(list(data.keys())[1:], num_params, replace=False)
            
            for param in params_to_anomalize:
                if param == "pH":
                    data[param][idx] = np.random.choice([3.5, 10.5])
                elif param == "Turbidity":
                    data[param][idx] = np.random.uniform(15, 25)
                elif param == "TDS":
                    data[param][idx] = np.random.uniform(800, 1000)
                elif param == "Temperature":
                    data[param][idx] = np.random.choice([1, 45])
                elif param == "Nitrate":
                    data[param][idx] = np.random.uniform(80, 120)
                elif param == "Chloride":
                    data[param][idx] = np.random.uniform(400, 600)
                elif param == "Fluoride":
                    data[param][idx] = np.random.uniform(3, 5)
                elif param == "Hardness":
                    data[param][idx] = np.random.uniform(500, 800)
                elif param == "DO":
                    data[param][idx] = np.random.uniform(0.5, 2)
                elif param == "BOD":
                    data[param][idx] = np.random.uniform(60, 100)
    
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
    
    # Color-code the dataframe based on quality and breaches
    def highlight_quality(row):
        # Check for breaches and anomalies
        has_breach = row.get('Breach_Count', 0) > 0
        has_anomaly = row.get('AI_Anomaly', 'Normal') == 'Anomaly'
        
        if has_breach or has_anomaly:
            return ['background-color: #ffcccc; color: black; font-weight: bold'] * len(row)  # Light red for issues
        elif row['Quality_Category'] == 'Excellent':
            return ['background-color: #d4edda; color: black'] * len(row)
        elif row['Quality_Category'] == 'Good':
            return ['background-color: #fff3cd; color: black'] * len(row)
        elif row['Quality_Category'] in ['Fair', 'Poor']:
            return ['background-color: #f8d7da; color: black'] * len(row)
        else:
            return ['background-color: #f5c6cb; color: black'] * len(row)
    
    # Display formatted data with additional warning indicators
    display_data = current_data.tail(20).copy()
    display_data['Timestamp'] = display_data['Timestamp'].dt.strftime('%H:%M:%S')
    
    # Add warning symbols to breach/anomaly rows
    display_data['Status'] = display_data.apply(lambda row: 
        '🚨 BREACH' if row.get('Breach_Count', 0) > 0 
        else '🤖 ANOMALY' if row.get('AI_Anomaly', 'Normal') == 'Anomaly'
        else '✅ NORMAL', axis=1)
    
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
    st.rerun()
