import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
import json
import math
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
    
    .status-excellent {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .status-good {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .status-fair {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .status-poor {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .wqi-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .treatment-module {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .cost-analysis {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .regulatory-compliance {
        background: #d1ecf1;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Functions ---
class WaterQualityDatabase:
    def __init__(self, db_name="water_quality.db"):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Main data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS water_quality_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                pH REAL,
                turbidity REAL,
                tds REAL,
                temperature REAL,
                nitrate REAL,
                chloride REAL,
                fluoride REAL,
                hardness REAL,
                dissolved_oxygen REAL,
                bod REAL,
                quality_score REAL,
                quality_category TEXT,
                wqi REAL,
                breach_count INTEGER,
                ai_anomaly TEXT,
                anomaly_score REAL,
                location TEXT DEFAULT 'Main Station',
                data_quality_flag TEXT DEFAULT 'Good'
            )
        ''')
        
        # Sensor calibration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor_type TEXT,
                calibration_date DATETIME,
                next_calibration DATETIME,
                calibration_status TEXT,
                drift_factor REAL DEFAULT 1.0
            )
        ''')
        
        # Treatment costs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS treatment_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATETIME,
                chemical_costs REAL,
                energy_costs REAL,
                maintenance_costs REAL,
                total_costs REAL,
                treatment_volume REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_data(self, data):
        """Insert water quality data"""
        conn = sqlite3.connect(self.db_name)
        data.to_sql('water_quality_data', conn, if_exists='append', index=False)
        conn.close()
    
    def get_data(self, limit=1000, start_date=None, end_date=None):
        """Retrieve water quality data"""
        conn = sqlite3.connect(self.db_name)
        
        query = "SELECT * FROM water_quality_data"
        params = []
        
        if start_date or end_date:
            query += " WHERE"
            if start_date:
                query += " timestamp >= ?"
                params.append(start_date)
            if end_date:
                if start_date:
                    query += " AND"
                query += " timestamp <= ?"
                params.append(end_date)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df

# --- Water Quality Index Calculations ---
class WQICalculator:
    def __init__(self):
        # WHO/EPA standards with weights
        self.standards = {
            "pH": {"ideal": 7.0, "min": 6.5, "max": 8.5, "weight": 0.12},
            "turbidity": {"ideal": 1.0, "min": 0, "max": 5, "weight": 0.08},
            "tds": {"ideal": 300, "min": 0, "max": 500, "weight": 0.10},
            "temperature": {"ideal": 20, "min": 5, "max": 30, "weight": 0.08},
            "nitrate": {"ideal": 10, "min": 0, "max": 45, "weight": 0.12},
            "chloride": {"ideal": 100, "min": 0, "max": 250, "weight": 0.10},
            "fluoride": {"ideal": 1.0, "min": 0, "max": 1.5, "weight": 0.08},
            "hardness": {"ideal": 150, "min": 0, "max": 300, "weight": 0.08},
            "dissolved_oxygen": {"ideal": 8, "min": 4, "max": 14, "weight": 0.14},
            "bod": {"ideal": 3, "min": 0, "max": 30, "weight": 0.10}
        }
    
    def calculate_wqi(self, data):
        """Calculate Water Quality Index"""
        wqi_scores = []
        
        for _, row in data.iterrows():
            total_weight = 0
            weighted_score = 0
            
            for param, standards in self.standards.items():
                if param in row and pd.notna(row[param]):
                    value = row[param]
                    
                    # Calculate sub-index
                    if value <= standards["min"]:
                        sub_index = 0
                    elif value >= standards["max"]:
                        sub_index = 0
                    else:
                        # Calculate based on distance from ideal
                        if value <= standards["ideal"]:
                            sub_index = 100 * (value - standards["min"]) / (standards["ideal"] - standards["min"])
                        else:
                            sub_index = 100 * (standards["max"] - value) / (standards["max"] - standards["ideal"])
                    
                    sub_index = max(0, min(100, sub_index))
                    
                    weighted_score += sub_index * standards["weight"]
                    total_weight += standards["weight"]
            
            if total_weight > 0:
                wqi = weighted_score / total_weight
            else:
                wqi = 0
            
            wqi_scores.append(round(wqi, 2))
        
        return wqi_scores
    
    def get_wqi_category(self, wqi):
        """Get WQI category"""
        if wqi >= 90:
            return "Excellent"
        elif wqi >= 70:
            return "Good"
        elif wqi >= 50:
            return "Fair"
        elif wqi >= 25:
            return "Poor"
        else:
            return "Unacceptable"

# --- Water Chemistry Calculations ---
class WaterChemistryCalculator:
    @staticmethod
    def langelier_saturation_index(ph, temperature, tds, hardness, alkalinity=None):
        """Calculate Langelier Saturation Index"""
        if alkalinity is None:
            alkalinity = hardness * 0.8  # Approximation
        
        # Temperature factor
        temp_factor = (2.6 / (1 + 0.01 * temperature)) - 1.8
        
        # TDS factor
        tds_factor = 0.12 * math.log10(tds)
        
        # Hardness factor
        hardness_factor = 0.34 * math.log10(hardness)
        
        # Alkalinity factor
        alkalinity_factor = 0.22 * math.log10(alkalinity)
        
        # pH saturation
        ph_saturation = 9.3 + temp_factor + tds_factor - hardness_factor - alkalinity_factor
        
        # LSI
        lsi = ph - ph_saturation
        
        return round(lsi, 2)
    
    @staticmethod
    def chlorine_demand(tds, temperature, ph, organic_matter=5):
        """Calculate chlorine demand"""
        base_demand = 0.5 + (tds / 1000) * 0.5
        temp_factor = 1 + (temperature - 20) * 0.02
        ph_factor = 1 + (ph - 7) * 0.1
        organic_factor = 1 + organic_matter * 0.05
        
        demand = base_demand * temp_factor * ph_factor * organic_factor
        return round(demand, 2)
    
    @staticmethod
    def membrane_fouling_potential(tds, turbidity, temperature):
        """Calculate membrane fouling potential"""
        sdi = (turbidity / 5) * 100  # Simplified SDI
        scaling_potential = (tds / 500) * 50
        temp_factor = max(1, (temperature - 25) * 2)
        
        fouling_potential = (sdi + scaling_potential) * temp_factor / 100
        return round(fouling_potential, 2)

# --- Treatment Process Simulator ---
class TreatmentSimulator:
    def __init__(self):
        self.coagulant_dose = 0
        self.chlorine_dose = 0
        self.energy_consumption = 0
        self.chemical_costs = 0
    
    def simulate_coagulation(self, turbidity, ph, temperature, alkalinity=100):
        """Simulate coagulation process"""
        # Optimal coagulant dose calculation
        base_dose = turbidity * 2.5  # mg/L alum per NTU
        
        # pH adjustment
        if ph < 6.5:
            ph_adjustment = (6.5 - ph) * 10
        elif ph > 7.5:
            ph_adjustment = (ph - 7.5) * 8
        else:
            ph_adjustment = 0
        
        # Temperature factor
        temp_factor = 1 + (20 - temperature) * 0.02
        
        optimal_dose = base_dose * temp_factor + ph_adjustment
        
        # Efficiency calculation
        efficiency = min(95, 85 + (optimal_dose / turbidity) * 2)
        
        # Treated water turbidity
        treated_turbidity = turbidity * (1 - efficiency / 100)
        
        return {
            'optimal_dose': round(optimal_dose, 2),
            'efficiency': round(efficiency, 1),
            'treated_turbidity': round(treated_turbidity, 2),
            'ph_adjustment_needed': round(ph_adjustment, 2)
        }
    
    def simulate_disinfection(self, temperature, ph, turbidity, contact_time=30):
        """Simulate disinfection process"""
        # Base chlorine demand
        base_demand = 0.5 + turbidity * 0.1
        
        # CT requirement (mg·min/L)
        ct_required = 0.5 * (1 + (ph - 7) * 0.2) * (1 + (temperature - 20) * 0.05)
        
        # Chlorine dose calculation
        chlorine_dose = base_demand + (ct_required / contact_time)
        
        # Log reduction
        log_reduction = min(6, contact_time * chlorine_dose / 10)
        
        return {
            'chlorine_dose': round(chlorine_dose, 2),
            'ct_value': round(chlorine_dose * contact_time, 2),
            'log_reduction': round(log_reduction, 1),
            'residual_chlorine': round(chlorine_dose * 0.3, 2)
        }
    
    def calculate_treatment_costs(self, flow_rate, coagulant_dose, chlorine_dose, energy_kwh):
        """Calculate treatment costs"""
        # Cost per unit (example prices in USD)
        coagulant_cost_per_kg = 0.8
        chlorine_cost_per_kg = 1.2
        energy_cost_per_kwh = 0.10
        
        # Daily costs
        coagulant_cost = (coagulant_dose / 1000) * flow_rate * 24 * coagulant_cost_per_kg
        chlorine_cost = (chlorine_dose / 1000) * flow_rate * 24 * chlorine_cost_per_kg
        energy_cost = energy_kwh * 24 * energy_cost_per_kwh
        
        total_cost = coagulant_cost + chlorine_cost + energy_cost
        
        return {
            'coagulant_cost': round(coagulant_cost, 2),
            'chlorine_cost': round(chlorine_cost, 2),
            'energy_cost': round(energy_cost, 2),
            'total_daily_cost': round(total_cost, 2),
            'cost_per_m3': round(total_cost / (flow_rate * 24), 4)
        }

# --- Enhanced Data Generation ---
def generate_realistic_data(n=100, location="Main Station", add_anomaly=False):
    """Generate realistic water quality data with correlations"""
    base_time = datetime.now()
    timestamps = [base_time - timedelta(minutes=i*10) for i in range(n)]
    timestamps.reverse()
    
    # Base parameters with realistic correlations
    temperature = 20 + 8 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 2, n)
    temperature = np.clip(temperature, 5, 35)
    
    # pH correlated with temperature and alkalinity
    ph_base = 7.2 + (temperature - 25) * 0.02 + np.random.normal(0, 0.3, n)
    ph = np.clip(ph_base, 6.0, 8.5)
    
    # Dissolved oxygen inversely correlated with temperature
    do_base = 12 - (temperature - 20) * 0.3 + np.random.normal(0, 1, n)
    dissolved_oxygen = np.clip(do_base, 3, 14)
    
    # TDS with seasonal variation
    tds_base = 350 + 50 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 40, n)
    tds = np.clip(tds_base, 100, 600)
    
    # Turbidity with weather events
    turbidity_base = np.random.lognormal(0.5, 0.8, n)
    # Add occasional high turbidity events (rainfall simulation)
    for i in range(n):
        if np.random.random() < 0.05:  # 5% chance of rainfall event
            turbidity_base[i] *= np.random.uniform(3, 8)
    turbidity = np.clip(turbidity_base, 0.1, 20)
    
    # Other parameters with realistic ranges
    nitrate = np.random.lognormal(2.0, 0.8, n)
    nitrate = np.clip(nitrate, 0.1, 50)
    
    chloride = 150 + (tds - 350) * 0.3 + np.random.normal(0, 30, n)
    chloride = np.clip(chloride, 20, 300)
    
    fluoride = np.random.lognormal(0.2, 0.6, n)
    fluoride = np.clip(fluoride, 0.1, 2.0)
    
    hardness = 200 + (tds - 350) * 0.4 + np.random.normal(0, 40, n)
    hardness = np.clip(hardness, 50, 400)
    
    # BOD correlated with temperature and organic matter
    bod_base = 3 + (temperature - 20) * 0.1 + np.random.lognormal(1.0, 0.6, n)
    bod = np.clip(bod_base, 0.5, 40)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'pH': np.round(ph, 2),
        'turbidity': np.round(turbidity, 2),
        'tds': np.round(tds, 2),
        'temperature': np.round(temperature, 2),
        'nitrate': np.round(nitrate, 2),
        'chloride': np.round(chloride, 2),
        'fluoride': np.round(fluoride, 2),
        'hardness': np.round(hardness, 2),
        'dissolved_oxygen': np.round(dissolved_oxygen, 2),
        'bod': np.round(bod, 2),
        'location': location,
        'data_quality_flag': 'Good'
    })
    
    # Add severe anomalies if requested
    if add_anomaly:
        anomaly_indices = np.random.choice(n, size=max(5, n//10), replace=False)
        for idx in anomaly_indices:
            # Contamination event - multiple parameters affected
            data.loc[idx, 'turbidity'] *= np.random.uniform(5, 15)
            data.loc[idx, 'bod'] *= np.random.uniform(3, 8)
            data.loc[idx, 'nitrate'] *= np.random.uniform(2, 5)
            data.loc[idx, 'pH'] += np.random.uniform(-1.5, 1.5)
            data.loc[idx, 'dissolved_oxygen'] *= np.random.uniform(0.3, 0.7)
            data.loc[idx, 'data_quality_flag'] = 'Anomaly'
    
    return data

# --- Initialize Components ---
@st.cache_resource
def init_components():
    db = WaterQualityDatabase()
    wqi_calc = WQICalculator()
    chem_calc = WaterChemistryCalculator()
    treatment_sim = TreatmentSimulator()
    return db, wqi_calc, chem_calc, treatment_sim

# --- Session State Initialization ---
if "current_data" not in st.session_state:
    st.session_state.current_data = pd.DataFrame()
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

# Initialize components
db, wqi_calc, chem_calc, treatment_sim = init_components()

# --- Main Application ---
st.markdown('<h1 class="main-header">💧 Advanced Water Quality Engineering System</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.1rem; color: #666; margin-bottom: 2rem;">
    🔬 Real-time Monitoring • 🧪 Treatment Simulation • 💰 Cost Analysis • 📊 Regulatory Compliance
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.markdown("# 🎛️ System Control Panel")

# Main navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Dashboard", "📊 Data Analysis", "🧪 Treatment Simulation", "💰 Cost Analysis", "📋 Compliance Reports", "🗺️ Spatial Analysis"]
)

# Data generation controls
st.sidebar.markdown("### 📡 Data Generation")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔄 Generate Data", type="primary"):
        with st.spinner("Generating realistic data..."):
            data = generate_realistic_data(n=150)
            # Calculate WQI
            data['wqi'] = wqi_calc.calculate_wqi(data)
            data['quality_category'] = data['wqi'].apply(wqi_calc.get_wqi_category)
            
            # Store in session state and database
            st.session_state.current_data = data
            db.insert_data(data)
            st.success("✅ Data generated and stored!")

with col2:
    if st.button("⚠️ Add Anomalies"):
        with st.spinner("Simulating contamination events..."):
            data = generate_realistic_data(n=150, add_anomaly=True)
            data['wqi'] = wqi_calc.calculate_wqi(data)
            data['quality_category'] = data['wqi'].apply(wqi_calc.get_wqi_category)
            
            st.session_state.current_data = data
            db.insert_data(data)
            st.warning("⚠️ Contamination events simulated!")

# Load recent data if none exists
if st.session_state.current_data.empty:
    recent_data = db.get_data(limit=100)
    if not recent_data.empty:
        st.session_state.current_data = recent_data
    else:
        # Generate initial data
        initial_data = generate_realistic_data(n=100)
        initial_data['wqi'] = wqi_calc.calculate_wqi(initial_data)
        initial_data['quality_category'] = initial_data['wqi'].apply(wqi_calc.get_wqi_category)
        st.session_state.current_data = initial_data
        db.insert_data(initial_data)

current_data = st.session_state.current_data

# --- Page Routing ---
if page == "🏠 Dashboard":
    # Dashboard content
    if not current_data.empty:
        latest = current_data.iloc[-1]
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🌡️ Temperature", f"{latest['temperature']:.1f}°C", 
                     f"{latest['temperature'] - 20:.1f}°C")
        
        with col2:
            st.metric("⚗️ pH", f"{latest['pH']:.1f}", 
                     f"{latest['pH'] - 7:.1f}")
        
        with col3:
            st.metric("💧 TDS", f"{latest['tds']:.0f} mg/L", 
                     f"{latest['tds'] - 300:.0f}")
        
        with col4:
            st.metric("🌊 Turbidity", f"{latest['turbidity']:.1f} NTU", 
                     f"{latest['turbidity'] - 1:.1f}")
        
        with col5:
            st.metric("✅ WQI", f"{latest['wqi']:.0f}", 
                     f"{latest['quality_category']}")
        
        # WQI Display
        st.markdown("---")
        wqi_value = latest['wqi']
        wqi_category = latest['quality_category']
        
        # Color-coded WQI display
        if wqi_category == "Excellent":
            st.markdown(f'<div class="status-excellent">🏆 Water Quality Index: {wqi_value:.0f} - {wqi_category}</div>', 
                       unsafe_allow_html=True)
        elif wqi_category == "Good":
            st.markdown(f'<div class="status-good">✅ Water Quality Index: {wqi_value:.0f} - {wqi_category}</div>', 
                       unsafe_allow_html=True)
        elif wqi_category == "Fair":
            st.markdown(f'<div class="status-fair">⚠️ Water Quality Index: {wqi_value:.0f} - {wqi_category}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-poor">🚨 Water Quality Index: {wqi_value:.0f} - {wqi_category}</div>', 
                       unsafe_allow_html=True)
        
        # Real-time Charts
        st.markdown("## 📈 Real-time Parameter Monitoring")
        
        # Parameter selection
        params = ['pH', 'temperature', 'turbidity', 'tds', 'dissolved_oxygen', 'nitrate']
        selected_params = st.multiselect("Select parameters:", params, default=params[:4])
        
        if selected_params:
            fig = make_subplots(
                rows=len(selected_params), cols=1,
                subplot_titles=selected_params,
                shared_xaxes=True,
                vertical_spacing=0.03
            )
            
            colors = px.colors.qualitative.Set3
            
            for i, param in enumerate(selected_params):
                fig.add_trace(
                    go.Scatter(
                        x=current_data['timestamp'],
                        y=current_data[param],
                        mode='lines+markers',
                        name=param,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=3)
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(height=150*len(selected_params), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Data Analysis":
    st.markdown("## 📊 Advanced Data Analysis")
    
    if not current_data.empty:
        # Statistical Summary
        st.subheader("📈 Statistical Summary")
        
        # Select numeric columns
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        summary_
