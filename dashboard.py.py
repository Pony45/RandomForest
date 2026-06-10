import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="M&V Dashboard", layout="wide")
st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Predict energy savings for Malaysian residential buildings*")

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model_path = 'models/thesis_mv_random_forest.pkl'
    features_path = 'models/thesis_mv_features.txt'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        return None, None
    
    if not os.path.exists(features_path):
        st.error(f"❌ Features file not found at {features_path}")
        return None, None
    
    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

# ==========================================
# SCALING FACTOR untuk rumah Malaysia
# Model original output terlalu besar (industrial scale)
# Kita scale down supaya realistik untuk residential
# ==========================================
SCALING_FACTOR = 10  # Bahagi 10 untuk dapatkan kWh untuk rumah biasa

def scale_prediction(prediction):
    """Scale down prediction for Malaysian residential buildings"""
    return prediction / SCALING_FACTOR

# ==========================================
# UNIT CONVERSION FUNCTION
# ==========================================
def convert_energy_unit(prediction_kwh, target_unit):
    if target_unit == "Per Hour (kWh)":
        return prediction_kwh, "kWh"
    elif target_unit == "Per Day (kWh)":
        return prediction_kwh * 24, "kWh/day"
    elif target_unit == "Per Month (kWh)":
        return prediction_kwh * 24 * 30, "kWh/month"
    elif target_unit == "Per Year (kWh)":
        return prediction_kwh * 24 * 365, "kWh/year"

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📋 Building Parameters")

# Unit selector
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Display Settings")
unit_option = st.sidebar.selectbox(
    "Energy Unit Display",
    ["Per Hour (kWh)", "Per Day (kWh)", "Per Month (kWh)", "Per Year (kWh)"]
)

# Model Performance
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")

with st.sidebar.expander("Performance Metrics", expanded=True):
    # Nilai dari training (sudah discale)
    r2_score = 0.8723
    mae = 0.12  # Scaled MAE
    rmse = 0.16  # Scaled RMSE
    
    st.metric("R² Score", f"{r2_score:.4f}")
    st.metric("MAE", f"{mae:.2f} kWh")
    st.caption(f"RMSE: {rmse:.2f} kWh")
    st.progress(r2_score, text=f"Accuracy: {r2_score*100:.1f}%")
    
    st.caption("📌 *Values scaled for Malaysian residential context*")

st.sidebar.markdown("---")

# Inputs
temp = st.sidebar.slider("🌡️ Temperature (°C)", 22, 35, 28)  # Range Malaysia
humidity = st.sidebar.slider("💧 Humidity (%)", 60, 95, 80)
hour = st.sidebar.slider("⏰ Hour of Day", 0, 23, 14)
dayofweek = st.sidebar.selectbox("📅 Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month = st.sidebar.selectbox("📆 Month", list(range(1,13)), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
floor_area = st.sidebar.number_input("🏠 Floor Area (m²)", 50, 300, 120)  # Rumah Malaysia 120m²
occupants = st.sidebar.number_input("👥 Occupants", 2, 8, 4)
retrofit = st.sidebar.selectbox("🔧 Retrofit Status", [0,1], format_func=lambda x: "✅ Yes (Retrofitted)" if x else "❌ No (Baseline)")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
is_weekend = 1 if dayofweek >= 5 else 0
temp_humidity = temp * humidity / 100
occ_per_area = occupants / floor_area

features_df = pd.DataFrame([[
    temp, humidity, hour, dayofweek, month, floor_area, occupants, retrofit,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend, temp_humidity, occ_per_area
]], columns=FEATURES)

# ==========================================
# MAIN CONTENT
# ==========================================
st.info("📌 **Note:** Energy values have been scaled for Malaysian residential context (typical home: 300-600 kWh/month)")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy", type="primary", use_container_width=True):
        # Get raw prediction and scale
        raw_pred = model.predict(features_df)[0]
        pred = scale_prediction(raw_pred)
        
        converted, unit = convert_energy_unit(pred, unit_option)
        
        st.subheader("📊 Prediction Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("⚡ Predicted Energy", f"{converted:.2f} {unit}")
        
        if retrofit == 1:
            # Baseline (without retrofit)
            base_df = features_df.copy()
            base_df['retrofit'] = 0
            raw_base = model.predict(base_df)[0]
            base_pred = scale_prediction(raw_base)
            
            savings = base_pred - pred
            savings_pct = (savings / base_pred) * 100
            savings_conv, _ = convert_energy_unit(savings, unit_option)
            
            m2.metric("💰 Savings", f"{savings_conv:.2f} {unit}", delta=f"{savings_pct:.1f}%")
            m3.metric("🏆 Reduction", f"{savings_pct:.1f}%", delta="Good!")
            
            st.success(f"💡 Retrofit Savings: {savings_conv:.2f} {unit} ({savings_pct:.1f}%)")
            
            # Calculate monthly bill savings (RM)
            tariff = 0.52  # TNB tariff ~ RM0.52/kWh
            monthly_savings = savings * 24 * 30  # Convert hourly to monthly
            monthly_rm = monthly_savings * tariff
            
            st.info(f"💰 **Estimated Monthly Bill Savings:** RM {monthly_rm:.2f}/month (at RM0.52/kWh)")
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(8,5))
            bars = ax.bar(['Baseline\n(No Retrofit)', 'Retrofitted\n(With Retrofit)'], 
                         [base_pred, pred], color=['#e74c3c', '#2ecc71'], edgecolor='black')
            
            for bar, val in zip(bars, [base_pred, pred]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f} kWh', ha='center', fontweight='bold')
            
            ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
            ax.set_title('Retrofit Impact on Energy Consumption', fontweight='bold', fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            # Gauge chart
            fig2, ax2 = plt.subplots(figsize=(8,2.5))
            
            if savings_pct < 10:
                color, label = '#e74c3c', 'Low Savings'
            elif savings_pct < 20:
                color, label = '#f39c12', 'Medium Savings'
            else:
                color, label = '#2ecc71', 'High Savings'
            
            ax2.barh([0], [min(savings_pct,100)], color=color, height=0.4, edgecolor='black')
            ax2.barh([0], [100], color='lightgray', height=0.4, alpha=0.3)
            ax2.set_xlim(0, 100)
            ax2.set_yticks([])
            ax2.set_xlabel('Energy Savings (%)', fontsize=11)
            ax2.set_title(f'Retrofit Efficiency: {label} ({savings_pct:.1f}% savings)', fontweight='bold')
            ax2.text(savings_pct + 2, 0, f'{savings_pct:.1f}%', va='center', fontweight='bold')
            ax2.axvline(x=10, color='orange', linestyle='--', alpha=0.5)
            ax2.axvline(x=20, color='green', linestyle='--', alpha=0.5)
            st.pyplot(fig2)
            
            # Savings summary table
            with st.expander("📋 Detailed Savings Summary", expanded=False):
                daily_savings = savings * 24
                monthly_savings = daily_savings * 30
                yearly_savings = monthly_savings * 12
                
                summary_data = {
                    'Period': ['Hour', 'Day', 'Month', 'Year'],
                    'Savings (kWh)': [f"{savings:.2f}", f"{daily_savings:.2f}", f"{monthly_savings:.2f}", f"{yearly_savings:.2f}"],
                    'Cost Savings (RM)': [
                        f"RM {savings * tariff:.2f}",
                        f"RM {daily_savings * tariff:.2f}",
                        f"RM {monthly_savings * tariff:.2f}",
                        f"RM {yearly_savings * tariff:.2f}"
                    ]
                }
                st.table(pd.DataFrame(summary_data))
            
        else:
            # Not retrofitted - show potential
            retro_df = features_df.copy()
            retro_df['retrofit'] = 1
            raw_retro = model.predict(retro_df)[0]
            retro_pred = scale_prediction(raw_retro)
            
            potential = pred - retro_pred
            potential_pct = (potential / pred) * 100
            potential_conv, _ = convert_energy_unit(potential, unit_option)
            
            m2.metric("💰 Potential Savings", f"{potential_conv:.2f} {unit}", delta=f"{potential_pct:.1f}%")
            m3.metric("🏆 Would Save", f"{potential_pct:.1f}%", delta="If retrofitted")
            
            st.info(f"💡 **If you retrofit this building:** Would save ~{potential_conv:.2f} {unit} ({potential_pct:.1f}%)")
            
            # Monthly savings estimate
            monthly_savings_est = potential * 24 * 30
            monthly_rm_est = monthly_savings_est * 0.52
            st.info(f"💰 **Estimated monthly bill savings after retrofit:** RM {monthly_rm_est:.2f}/month")
            
            st.caption("👉 **Tip:** Select 'Yes (Retrofitted)' above to see detailed savings analysis with graphs!")
            
            # Simple comparison chart
            fig_simple, ax_simple = plt.subplots(figsize=(8, 5))
            ax_simple.bar(['Current\n(No Retrofit)', 'If Retrofitted'], 
                         [pred, retro_pred], color=['#e74c3c', '#2ecc71'], edgecolor='black')
            
            for i, v in enumerate([pred, retro_pred]):
                ax_simple.text(i, v + 0.05, f'{v:.2f} kWh', ha='center', fontweight='bold')
            
            ax_simple.set_ylabel('Energy Consumption (kWh)', fontsize=12)
            ax_simple.set_title('Potential Impact of Retrofit', fontweight='bold', fontsize=14)
            ax_simple.grid(axis='y', alpha=0.3)
            st.pyplot(fig_simple)

with col2:
    st.info("""
    **📖 About this M&V System**
    
    🎯 **Model:** Random Forest Regressor
    
    📊 **Features:**
    - Weather (temp, humidity)
    - Time (hour, day, month)
    - Building (area, occupants)
    - Retrofit status
    
    🏠 **Malaysia Context:**
    - Scaled for residential homes
    - Typical home: 300-600 kWh/month
    - TNB tariff: ~RM0.52/kWh
    
    💡 **How to use:**
    1. Adjust building parameters
    2. Click "Predict"
    3. See potential savings
    """)

st.markdown("---")
st.caption("🎓 AI-based M&V System | Thesis Project | Scaled for Malaysian Residential Buildings")
