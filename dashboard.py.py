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
    """Convert hourly kWh to different time units"""
    if target_unit == "Per Hour (kWh)":
        return prediction_kwh, "kWh"
    elif target_unit == "Per Day (kWh)":
        return prediction_kwh * 24, "kWh/day"
    elif target_unit == "Per Month (kWh)":
        return prediction_kwh * 24 * 30, "kWh/month"
    elif target_unit == "Per Year (kWh)":
        return prediction_kwh * 24 * 365, "kWh/year"

# ==========================================
# SIDEBAR - PARAMETERS & SETTINGS
# ==========================================
st.sidebar.header("📋 Building Parameters")

# Unit selector (letak dulu)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Display Settings")
unit_option = st.sidebar.selectbox(
    "Energy Unit Display",
    ["Per Hour (kWh)", "Per Day (kWh)", "Per Month (kWh)", "Per Year (kWh)"],
    help="Convert prediction to different time units"
)

# Model Performance Metrics
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")

with st.sidebar.expander("Performance Metrics", expanded=True):
    # ⚠️ GANTI NILAI INI DENGAN NILAI DARI TRAINING MODEL 1 ⚠️
    r2_score = 0.8723   # ← Ganti dengan R² Score sebenar
    mae = 1.23          # ← Ganti dengan MAE sebenar (kWh)
    rmse = 1.56         # ← Ganti dengan RMSE sebenar (kWh)
    
    st.metric("R² Score", f"{r2_score:.4f}", 
              help="Higher is better (1.0 = perfect prediction)")
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} kWh", 
              help="Lower is better - average prediction error")
    st.caption(f"RMSE: {rmse:.2f} kWh")
    st.progress(min(r2_score, 1.0), text=f"Model Accuracy: {r2_score*100:.1f}%")

st.sidebar.markdown("---")

# Input parameters
temp = st.sidebar.slider("🌡️ Temperature (°C)", -5, 45, 22)
humidity = st.sidebar.slider("💧 Humidity (%)", 20, 100, 60)
hour = st.sidebar.slider("⏰ Hour of Day", 0, 23, 14)
dayofweek = st.sidebar.selectbox("📅 Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][x])
month = st.sidebar.selectbox("📆 Month", list(range(1,13)), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
floor_area = st.sidebar.number_input("🏠 Floor Area (m²)", 30, 300, 90)
occupants = st.sidebar.number_input("👥 Number of Occupants", 1, 10, 3)
retrofit = st.sidebar.selectbox("🔧 Retrofit Status", [0,1], format_func=lambda x: "✅ Yes (Retrofitted)" if x==1 else "❌ No (Baseline)")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
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

# ==========================================
# MAIN CONTENT
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
        prediction = model.predict(features_df)[0]
        
        # Convert based on user selection
        converted_value, unit_label = convert_energy_unit(prediction, unit_option)
        
        # Display metrics
        st.subheader("📊 Prediction Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("⚡ Predicted Energy", f"{converted_value:.2f} {unit_label}")
        
        if retrofit == 1:
            # Calculate baseline (without retrofit)
            features_baseline = features_df.copy()
            features_baseline['retrofit'] = 0
            baseline_pred = model.predict(features_baseline)[0]
            
            # Calculate savings
            savings = baseline_pred - prediction
            savings_pct = (savings / baseline_pred) * 100
            
            # Convert savings
            savings_converted, _ = convert_energy_unit(savings, unit_option)
            
            m2.metric("💰 Energy Savings", f"{savings_converted:.2f} {unit_label}", delta=f"{savings_pct:.1f}%")
            m3.metric("🏆 Efficiency Gain", f"{savings_pct:.1f}%", delta="Reduction")
            
            st.success(f"💡 **Retrofit Savings:** {savings_converted:.2f} {unit_label} ({savings_pct:.1f}% less energy)")
            
            # ==========================================
            # GRAPH 1: BAR CHART
            # ==========================================
            st.subheader("📊 Energy Consumption Comparison")
            
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            categories = ['Baseline\n(No Retrofit)', 'Retrofitted\n(With Retrofit)']
            values = [baseline_pred, prediction]
            
            bars = ax1.bar(categories, values, color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{val:.2f} kWh', ha='center', fontweight='bold', fontsize=11)
            
            ax1.annotate(f'💡 Savings: {savings:.2f} kWh\n({savings_pct:.1f}%)',
                        xy=(1, prediction + savings/2),
                        xytext=(1.3, prediction + savings/2 + 1),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            
            ax1.set_ylabel('Energy Consumption (kWh)', fontsize=12)
            ax1.set_title('Retrofit Impact on Energy Consumption', fontweight='bold', fontsize=14)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            ax1.set_ylim(0, max(values) + 2)
            st.pyplot(fig1)
            
            # ==========================================
            # GRAPH 2: DONUT CHART
            # ==========================================
            st.subheader("📊 Savings Breakdown")
            
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            sizes = [savings, baseline_pred - savings]
            labels = [f'Savings\n({savings_pct:.1f}%)', f'Remaining Energy\n({100-savings_pct:.1f}%)']
            
            ax2.pie(sizes, labels=labels, colors=['#2ecc71', '#e74c3c'],
                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            
            centre_circle = plt.Circle((0,0), 0.70, fc='white', linewidth=2)
            fig2.gca().add_artist(centre_circle)
            ax2.set_title('Energy Savings Distribution', fontweight='bold', fontsize=14)
            st.pyplot(fig2)
            
            # ==========================================
            # GRAPH 3: GAUGE CHART
            # ==========================================
            st.subheader("📊 Efficiency Score")
            
            fig3, ax3 = plt.subplots(figsize=(8, 2))
            efficiency_score = min(savings_pct, 100)
            
            if efficiency_score < 30:
                color, label = '#e74c3c', 'Low Efficiency'
            elif efficiency_score < 60:
                color, label = '#f39c12', 'Medium Efficiency'
            else:
                color, label = '#2ecc71', 'High Efficiency'
            
            ax3.barh([0], [efficiency_score], color=color, height=0.4, edgecolor='black')
            ax3.barh([0], [100], color='lightgray', height=0.4, alpha=0.3)
            ax3.set_xlim(0, 100)
            ax3.set_yticks([])
            ax3.set_xlabel('Efficiency Score (%)', fontsize=10)
            ax3.set_title(f'Retrofit Efficiency: {label} ({efficiency_score:.1f}% savings)', fontweight='bold')
            ax3.text(efficiency_score + 2, 0, f'{efficiency_score:.1f}%', va='center', fontweight='bold')
            ax3.axvline(x=30, color='orange', linestyle='--', alpha=0.7)
            ax3.axvline(x=60, color='green', linestyle='--', alpha=0.7)
            ax3.text(15, -0.3, 'Poor', ha='center', fontsize=8)
            ax3.text(45, -0.3, 'Medium', ha='center', fontsize=8)
            ax3.text(80, -0.3, 'Good', ha='center', fontsize=8)
            st.pyplot(fig3)
            
            # ==========================================
            # GRAPH 4: SCENARIO ANALYSIS
            # ==========================================
            with st.expander("📈 Advanced Analysis: Savings by Factor"):
                st.markdown("**How different factors affect your savings:**")
                
                scenarios = ["Current"]
                savings_scenarios = [savings]
                
                # Hotter temperature
                features_hot = features_df.copy()
                features_hot['temperature'] = min(45, temp + 10)
                baseline_hot = model.predict(features_hot)[0]
                features_hot_retrofit = features_hot.copy()
                features_hot_retrofit['retrofit'] = 1
                retrofit_hot = model.predict(features_hot_retrofit)[0]
                scenarios.append(f"+10°C\n({temp+10:.0f}°C)")
                savings_scenarios.append(baseline_hot - retrofit_hot)
                
                # Peak hour
                features_peak = features_df.copy()
                features_peak['hour'] = 18
                baseline_peak = model.predict(features_peak)[0]
                features_peak_retrofit = features_peak.copy()
                features_peak_retrofit['retrofit'] = 1
                retrofit_peak = model.predict(features_peak_retrofit)[0]
                scenarios.append("Peak Hour\n(6pm)")
                savings_scenarios.append(baseline_peak - retrofit_peak)
                
                # More occupants
                features_more = features_df.copy()
                features_more['occupants'] = min(10, occupants + 3)
                baseline_more = model.predict(features_more)[0]
                features_more_retrofit = features_more.copy()
                features_more_retrofit['retrofit'] = 1
                retrofit_more = model.predict(features_more_retrofit)[0]
                scenarios.append(f"+3 Occupants\n({occupants+3})")
                savings_scenarios.append(baseline_more - retrofit_more)
                
                fig4, ax4 = plt.subplots(figsize=(10, 5))
                bars4 = ax4.bar(scenarios, savings_scenarios, color='#3498db', edgecolor='black')
                
                for bar, val in zip(bars4, savings_scenarios):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{val:.2f} kWh', ha='center', fontweight='bold', fontsize=9)
                
                ax4.set_ylabel('Potential Savings (kWh)', fontsize=12)
                ax4.set_title('Retrofit Savings Under Different Conditions', fontweight='bold', fontsize=14)
                ax4.grid(axis='y', alpha=0.3)
                st.pyplot(fig4)
            
            # Comparison table
            with st.expander("📋 Detailed Comparison Table"):
                comparison_data = {
                    'Metric': ['Energy Consumption', 'Savings', 'Efficiency'],
                    'Baseline (No Retrofit)': [f"{baseline_pred:.2f} kWh", '-', '-'],
                    'Retrofitted': [f"{prediction:.2f} kWh", f"{savings:.2f} kWh", f"{savings_pct:.1f}%"]
                }
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                
        else:
            # Not retrofitted - show potential
            features_retrofit = features_df.copy()
            features_retrofit['retrofit'] = 1
            retrofit_pred = model.predict(features_retrofit)[0]
            potential_savings = prediction - retrofit_pred
            potential_pct = (potential_savings / prediction) * 100
            
            potential_converted, _ = convert_energy_unit(potential_savings, unit_option)
            
            m2.metric("💰 Potential Savings", f"{potential_converted:.2f} {unit_label}", delta=f"{potential_pct:.1f}%")
            m3.metric("🏆 Would Save", f"{potential_pct:.1f}%", delta="If retrofitted")
            
            st.info(f"💡 **If you retrofit this building:** Would save ~{potential_converted:.2f} {unit_label} ({potential_pct:.1f}%)")
            st.caption("👉 **Tip:** Try selecting 'Yes (Retrofitted)' above to see actual savings with detailed graphs!")
            
            # Simple comparison graph
            fig_simple, ax_simple = plt.subplots(figsize=(8, 5))
            ax_simple.bar(['Current\n(No Retrofit)', 'If Retrofitted'], [prediction, retrofit_pred],
                         color=['#e74c3c', '#2ecc71'], edgecolor='black')
            for i, v in enumerate([prediction, retrofit_pred]):
                ax_simple.text(i, v + 0.2, f'{v:.2f} kWh', ha='center', fontweight='bold')
            ax_simple.set_ylabel('Energy Consumption (kWh)')
            ax_simple.set_title('Potential Impact of Retrofit', fontweight='bold', fontsize=14)
            ax_simple.grid(axis='y', alpha=0.3)
            st.pyplot(fig_simple)

with col2:
    st.info("""
    **📖 About this M&V System**
    
    🎯 **Model:** Random Forest Regressor
    
    📊 **Features used:**
    - Weather (temp, humidity)
    - Time (hour, day, month)
    - Building (area, occupants)
    - Retrofit status
    
    💡 **How to use:**
    1. Adjust building parameters
    2. Click "Predict"
    3. Compare baseline vs retrofit
    
    ✅ **Interpret graphs:**
    - **Bar chart:** Energy comparison
    - **Donut chart:** Savings breakdown  
    - **Gauge:** Efficiency score
    - **Scenario analysis:** Compare conditions
    
    📈 **Model Performance:**
    - R² = {:.4f}
    - MAE = {:.2f} kWh
    """.format(r2_score, mae) if 'r2_score' in dir() else "")
    
    st.markdown("---")
    st.caption("🎓 AI-based Measurement & Verification System | Thesis Project")

st.markdown("---")

# Footer with interpretation guide
with st.expander("📖 How to Interpret the Graphs", expanded=False):
    st.markdown("""
    ### 📊 Graph Interpretation Guide
    
    | Graph | What it shows | How to use |
    |-------|---------------|------------|
    | **Bar Chart** | Baseline vs Retrofitted energy | Bigger difference = more savings |
    | **Donut Chart** | What percentage of energy is saved | Target: >20% savings |
    | **Efficiency Gauge** | How effective the retrofit is | Green = Good, Red = Poor |
    | **Scenario Analysis** | Savings under different conditions | Identifies best retrofit conditions |
    
    ### 🎯 Key Insights
    - **Higher savings** occur during extreme temperatures (hot/cold)
    - **Peak hours** (morning 7-9am, evening 5-7pm) show higher savings
    - **Larger buildings** with more occupants benefit more from retrofit
    """)
