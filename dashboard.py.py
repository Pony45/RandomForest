import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="M&V Dashboard", layout="wide")
st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Predict energy savings from residential building retrofits using Random Forest*")

# Load model
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

st.sidebar.header("📋 Building Parameters")

temp = st.sidebar.slider("Temperature (°C)", -5, 45, 22)
humidity = st.sidebar.slider("Humidity (%)", 20, 100, 60)
hour = st.sidebar.slider("Hour of Day", 0, 23, 14)
dayofweek = st.sidebar.selectbox("Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month = st.sidebar.selectbox("Month", list(range(1,13)))
floor_area = st.sidebar.number_input("Floor Area (m²)", 30, 300, 90)
occupants = st.sidebar.number_input("Number of Occupants", 1, 10, 3)
retrofit = st.sidebar.selectbox("Retrofit Status", [0,1], format_func=lambda x: "✅ Yes (Retrofitted)" if x==1 else "❌ No (Baseline)")

# Feature engineering
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
is_weekend = 1 if dayofweek >= 5 else 0
temp_humidity_interaction = temp * humidity / 100
occupants_per_area = occupants / floor_area

# Prepare features
features_df = pd.DataFrame([[
    temp, humidity, hour, dayofweek, month, floor_area, occupants, retrofit,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend,
    temp_humidity_interaction, occupants_per_area
]], columns=FEATURES)

col1, col2 = st.columns(2)

with col1:
    if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
        prediction = model.predict(features_df)[0]
        
        st.metric("⚡ Predicted Energy", f"{prediction:.2f} kWh")
        
        # ==========================================
        # BETULKAN SAVINGS: Kira baseline vs retrofit
        # ==========================================
        if retrofit == 1:
            # Kira baseline (seolah-olah tak retrofit)
            features_baseline = features_df.copy()
            features_baseline['retrofit'] = 0
            baseline_pred = model.predict(features_baseline)[0]
            
            # Kira savings
            savings = baseline_pred - prediction
            savings_pct = (savings / baseline_pred) * 100
            
            st.success(f"💡 Retrofit Savings: {savings:.2f} kWh ({savings_pct:.1f}%)")
            
            # Optional: Tunjukkan perbandingan
            with st.expander("📊 View Comparison"):
                col_a, col_b = st.columns(2)
                col_a.metric("Baseline (No Retrofit)", f"{baseline_pred:.2f} kWh")
                col_b.metric("Retrofitted", f"{prediction:.2f} kWh", delta=f"-{savings:.2f} kWh")
        else:
            # Kalau retrofit = 0, tunjuk potential savings kalau retrofit
            features_retrofit = features_df.copy()
            features_retrofit['retrofit'] = 1
            retrofit_pred = model.predict(features_retrofit)[0]
            potential_savings = prediction - retrofit_pred
            potential_pct = (potential_savings / prediction) * 100
            
            st.info(f"💡 If retrofitted: Would save ~{potential_savings:.2f} kWh ({potential_pct:.1f}%)")
            st.caption("👉 Try selecting 'Yes (Retrofitted)' to see actual savings")

with col2:
    st.info("""
    **📖 About this M&V System**
    - Random Forest Regressor model
    - Trained on residential building data
    - Predicts hourly energy consumption
    - Estimates retrofit savings
    """)
    # Dalam dashboard.py, lepas prediction, tambah ni:

if st.button("Predict"):
    # ... existing code ...
    
    # Bar chart comparison
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    bars = ax.bar(['Baseline', 'Retrofitted'], [baseline_pred, prediction], 
                  color=['red', 'green'])
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Energy Consumption Comparison')
    
    for bar, val in zip(bars, [baseline_pred, prediction]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center')
    
    st.pyplot(fig)

st.markdown("---")
st.caption("🎓 AI-based Measurement & Verification System | Thesis Project")
