import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="M&V Dashboard", layout="wide")

st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Energy savings prediction for residential building retrofits using Random Forest*")

# Load model
@st.cache_resource
def load_model():
    model_path = 'models/thesis_mv_random_forest.pkl'
    features_path = 'models/thesis_mv_features.txt'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = [line.strip() for line in f.readlines()]
    else:
        features = ['floor_area', 'building_age', 'floors', 'retrofit', 'region_code', 'class_code']
    
    return model, features

# Load metrics
@st.cache_resource
def load_metrics():
    metrics_path = 'models/model_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

model, FEATURES = load_model()
metrics = load_metrics()

if model is None:
    st.stop()

st.success("✅ Model loaded successfully!")

# Show metrics in sidebar
if metrics:
    with st.sidebar.expander("📊 Model Performance", expanded=True):
        st.metric("R² Score", f"{metrics['r2_score']:.4f}")
        st.metric("MAE", f"{metrics['mae']:.2f} kWh/m²/yr")
        st.caption(f"RMSE: {metrics['rmse']:.2f} kWh/m²/yr")

# Sidebar inputs
st.sidebar.header("📋 Building Parameters")

floor_area = st.sidebar.number_input("🏠 Floor Area (m²)", 30, 300, 90)
building_age = st.sidebar.number_input("📅 Building Age (years)", 0, 150, 20)
floors = st.sidebar.selectbox("🏢 Number of Floors", [1, 2, 3, 4, 5])
retrofit = st.sidebar.selectbox("🔧 Retrofit Status", [0, 1], format_func=lambda x: "✅ Yes (Retrofitted)" if x else "❌ No (Baseline)")
region_code = st.sidebar.selectbox("📍 Region", [0, 1, 2, 3, 4, 5], format_func=lambda x: ["Riga", "Liepaja", "Ventspils", "Jelgava", "Jurmala", "Other"][x])
class_code = st.sidebar.selectbox("📊 Energy Class", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["A", "B", "C", "D", "E", "F", "G"][x])

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
        # Prepare features
        data = [[floor_area, building_age, floors, retrofit, region_code, class_code]]
        df_input = pd.DataFrame(data, columns=FEATURES)
        prediction = model.predict(df_input)[0]
        
        # Calculate totals
        total_per_year = prediction * floor_area
        total_per_month = total_per_year / 12
        total_per_day = total_per_year / 365
        
        # Display metrics
        st.subheader("📊 Prediction Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⚡ Per m² per Year", f"{prediction:.1f} kWh/m²/yr")
        m2.metric("🏠 Total per Year", f"{total_per_year:.0f} kWh/yr")
        m3.metric("📅 Total per Month", f"{total_per_month:.0f} kWh/month")
        m4.metric("🌙 Total per Day", f"{total_per_day:.1f} kWh/day")
        
        if retrofit == 1:
            # Calculate baseline
            data_base = [[floor_area, building_age, floors, 0, region_code, class_code]]
            baseline_pred = model.predict(pd.DataFrame(data_base, columns=FEATURES))[0]
            baseline_total = baseline_pred * floor_area
            
            savings = baseline_pred - prediction
            savings_pct = (savings / baseline_pred) * 100
            savings_total = savings * floor_area
            
            st.subheader("📊 Energy Savings Analysis")
            
            # Bar chart
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            categories = ['Baseline\n(No Retrofit)', 'Retrofitted\n(With Retrofit)']
            values = [baseline_pred, prediction]
            
            ax1.bar(categories, values, color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
            for i, v in enumerate(values):
                ax1.text(i, v + 1, f'{v:.1f} kWh/m²/yr', ha='center', fontweight='bold')
            
            ax1.annotate(f'💡 Savings: {savings:.1f} kWh/m²/yr ({savings_pct:.1f}%)',
                        xy=(1, prediction + savings/2), xytext=(1.3, prediction + savings/2 + 5),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
            
            ax1.set_ylabel('Energy Consumption (kWh/m²/year)')
            ax1.set_title('Retrofit Impact on Energy Consumption', fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            st.pyplot(fig1)
            
            # Savings gauge
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            if savings_pct < 10:
                color = '#e74c3c'
                label = 'Low Savings'
            elif savings_pct < 20:
                color = '#f39c12'
                label = 'Medium Savings'
            else:
                color = '#2ecc71'
                label = 'High Savings'
            
            ax2.barh([0], [savings_pct], color=color, height=0.4, edgecolor='black')
            ax2.barh([0], [100], color='lightgray', height=0.4, alpha=0.3)
            ax2.set_xlim(0, 100)
            ax2.set_yticks([])
            ax2.set_xlabel('Energy Savings (%)')
            ax2.set_title(f'Retrofit Efficiency: {label} ({savings_pct:.1f}% savings)', fontweight='bold')
            ax2.text(savings_pct + 2, 0, f'{savings_pct:.1f}%', va='center', fontweight='bold')
            ax2.axvline(x=10, color='orange', linestyle='--', alpha=0.5)
            ax2.axvline(x=20, color='green', linestyle='--', alpha=0.5)
            st.pyplot(fig2)
            
            # Total energy comparison
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.bar(['Baseline\n(No Retrofit)', 'Retrofitted'], [baseline_total, total_per_year],
                   color=['#e74c3c', '#2ecc71'], edgecolor='black')
            for i, v in enumerate([baseline_total, total_per_year]):
                ax3.text(i, v + 50, f'{v:,.0f} kWh', ha='center', fontweight='bold')
            ax3.set_ylabel('Total Energy Consumption (kWh/year)')
            ax3.set_title('Total Annual Energy Consumption', fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            st.pyplot(fig3)
            
            # Savings summary
            st.info(f"""
            ### 💰 Retrofit Savings Summary
            | Metric | Value |
            |--------|-------|
            | **Energy Savings** | {savings:.1f} kWh/m²/year |
            | **Percentage Reduction** | {savings_pct:.1f}% |
            | **Total Annual Savings** | {savings_total:,.0f} kWh/year |
            """)
            
        else:
            # Show potential savings
            data_retro = [[floor_area, building_age, floors, 1, region_code, class_code]]
            retrofit_pred = model.predict(pd.DataFrame(data_retro, columns=FEATURES))[0]
            potential = prediction - retrofit_pred
            potential_pct = (potential / prediction) * 100
            
            st.info(f"""
            💡 **If you retrofit this building:**
            - Would save ~{potential:.1f} kWh/m²/year ({potential_pct:.1f}%)
            - **Try selecting 'Retrofitted' above to see full analysis!**
            """)
            
            # Simple comparison chart
            fig_simple, ax_simple = plt.subplots(figsize=(8, 5))
            ax_simple.bar(['Current\n(No Retrofit)', 'If Retrofitted'], [prediction, retrofit_pred],
                         color=['#e74c3c', '#2ecc71'], edgecolor='black')
            ax_simple.set_ylabel('Energy (kWh/m²/year)')
            ax_simple.set_title('Potential Retrofit Impact')
            st.pyplot(fig_simple)

with col2:
    st.info("""
    **📖 About this M&V System**
    
    | Item | Details |
    |------|---------|
    | **Model** | Random Forest Regressor |
    | **Dataset** | RETROFIT-LAT |
    | **Samples** | 1,010 buildings |
    | **Target** | kWh/m²/year |
    
    **Graphs Interpretation:**
    - **Bar Chart:** Compare baseline vs retrofit
    - **Gauge:** Savings efficiency
    - **Savings Summary:** Annual impact
    """)

st.markdown("---")
st.caption("🎓 AI-based Measurement & Verification System | Thesis Project")
