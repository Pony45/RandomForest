import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="M&V Dashboard", layout="wide")
st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Predict energy savings from residential building retrofits using Random Forest*")

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
    # ⚠️ TUKAR NILAI INI dengan nilai training sebenar
    r2_score = 0.8723
    mae = 1.23
    rmse = 1.56
    
    st.metric("R² Score", f"{r2_score:.4f}")
    st.metric("MAE", f"{mae:.2f} kWh")
    st.caption(f"RMSE: {rmse:.2f} kWh")
    st.progress(r2_score, text=f"Accuracy: {r2_score*100:.1f}%")

st.sidebar.markdown("---")

# Inputs
temp = st.sidebar.slider("🌡️ Temperature (°C)", -5, 45, 22)
humidity = st.sidebar.slider("💧 Humidity (%)", 20, 100, 60)
hour = st.sidebar.slider("⏰ Hour of Day", 0, 23, 14)
dayofweek = st.sidebar.selectbox("📅 Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month = st.sidebar.selectbox("📆 Month", list(range(1,13)), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
floor_area = st.sidebar.number_input("🏠 Floor Area (m²)", 30, 300, 90)
occupants = st.sidebar.number_input("👥 Occupants", 1, 10, 3)
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
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy", type="primary", use_container_width=True):
        pred = model.predict(features_df)[0]
        converted, unit = convert_energy_unit(pred, unit_option)
        
        st.subheader("📊 Prediction Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("⚡ Predicted Energy", f"{converted:.2f} {unit}")
        
        if retrofit == 1:
            # Baseline
            base_df = features_df.copy()
            base_df['retrofit'] = 0
            base_pred = model.predict(base_df)[0]
            
            savings = base_pred - pred
            savings_pct = (savings / base_pred) * 100
            savings_conv, _ = convert_energy_unit(savings, unit_option)
            
            m2.metric("💰 Savings", f"{savings_conv:.2f} {unit}", delta=f"{savings_pct:.1f}%")
            m3.metric("🏆 Reduction", f"{savings_pct:.1f}%", delta="Good!")
            
            st.success(f"💡 Retrofit Savings: {savings_conv:.2f} {unit} ({savings_pct:.1f}%)")
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(8,5))
            ax.bar(['Baseline', 'Retrofitted'], [base_pred, pred], color=['#e74c3c', '#2ecc71'], edgecolor='black')
            ax.text(0, base_pred+0.2, f'{base_pred:.1f} kWh', ha='center', fontweight='bold')
            ax.text(1, pred+0.2, f'{pred:.1f} kWh', ha='center', fontweight='bold')
            ax.set_ylabel('Energy (kWh)')
            ax.set_title('Retrofit Impact', fontweight='bold')
            st.pyplot(fig)
            
            # Gauge
            fig2, ax2 = plt.subplots(figsize=(8,2))
            color = '#2ecc71' if savings_pct > 20 else '#f39c12' if savings_pct > 10 else '#e74c3c'
            ax2.barh([0], [min(savings_pct,100)], color=color, height=0.4)
            ax2.barh([0], [100], color='lightgray', height=0.4, alpha=0.3)
            ax2.set_xlim(0,100)
            ax2.set_yticks([])
            ax2.set_xlabel('Savings (%)')
            ax2.set_title(f'Efficiency: {savings_pct:.1f}% savings')
            st.pyplot(fig2)
            
        else:
            # Not retrofitted
            retro_df = features_df.copy()
            retro_df['retrofit'] = 1
            retro_pred = model.predict(retro_df)[0]
            potential = pred - retro_pred
            potential_pct = (potential / pred) * 100
            potential_conv, _ = convert_energy_unit(potential, unit_option)
            
            m2.metric("💰 Potential Savings", f"{potential_conv:.2f} {unit}", delta=f"{potential_pct:.1f}%")
            m3.metric("🏆 Would Save", f"{potential_pct:.1f}%", delta="If retrofitted")
            
            st.info(f"💡 If retrofitted: Save ~{potential_conv:.2f} {unit} ({potential_pct:.1f}%)")
            st.caption("👉 Select 'Yes (Retrofitted)' to see detailed analysis")

with col2:
    st.info("""
    **📖 About**
    - **Model:** Random Forest
    - **Data:** Synthetic (demo)
    - **Features:** Temp, humidity, hour, day, month, area, occupants, retrofit
    
    **Unit conversion:**
    - Hour = original prediction
    - Day = Hour × 24
    - Month = Hour × 24 × 30
    - Year = Hour × 24 × 365
    """)

st.markdown("---")
st.caption("🎓 AI-based M&V System | Thesis Project")
