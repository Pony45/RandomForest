import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="M&V Dashboard", layout="wide")
st.title("🏠 Energy Savings M&V Dashboard")

@st.cache_resource
def load_model():
    model = joblib.load('models/thesis_mv_random_forest.pkl')
    with open('models/thesis_mv_features.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return model, features

try:
    model, FEATURES = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.sidebar.header("📋 Building Parameters")

temp = st.sidebar.slider("Temperature (°C)", -5, 45, 22)
humidity = st.sidebar.slider("Humidity (%)", 20, 100, 60)
hour = st.sidebar.slider("Hour of Day", 0, 23, 14)
floor_area = st.sidebar.number_input("Floor Area (m²)", 30, 300, 90)
occupants = st.sidebar.number_input("Number of Occupants", 1, 10, 3)
retrofit = st.sidebar.selectbox("Retrofit Status", ["No (Baseline)", "Yes (Retrofitted)"])

col1, col2 = st.columns(2)

with col1:
    if st.button("🔮 Predict Energy", type="primary", use_container_width=True):
        retrofit_val = 1 if retrofit == "Yes (Retrofitted)" else 0
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        features_df = pd.DataFrame([[
            temp, humidity, hour, 0, 6, floor_area, occupants, retrofit_val,
            hour_sin, hour_cos, 0, 0, 0, temp*humidity/100, occupants/floor_area
        ]], columns=FEATURES)
        
        prediction = model.predict(features_df)[0]
        
        st.metric("⚡ Predicted Energy Consumption", f"{prediction:.2f} kWh")
        
        if retrofit == "Yes (Retrofitted)":
            savings = prediction * 0.15
            st.success(f"💡 Estimated Savings: {savings:.2f} kWh ({savings/prediction*100:.1f}%)")

with col2:
    st.info("""
    **About this Dashboard**
    - Random Forest model for M&V
    - Predicts energy consumption
    - Estimates retrofit savings
    - Based on building parameters
    """)

st.markdown("---")
st.caption("🎓 AI-based Measurement & Verification System | Thesis Project")
