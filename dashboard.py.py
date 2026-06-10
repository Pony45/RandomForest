import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

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

temp = st.sidebar.slider("🌡️ Temperature (°C)", -5, 45, 22)
humidity = st.sidebar.slider("💧 Humidity (%)", 20, 100, 60)
hour = st.sidebar.slider("⏰ Hour of Day", 0, 23, 14)
dayofweek = st.sidebar.selectbox("📅 Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][x])
month = st.sidebar.selectbox("📆 Month", list(range(1,13)), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
floor_area = st.sidebar.number_input("🏠 Floor Area (m²)", 30, 300, 90)
occupants = st.sidebar.number_input("👥 Number of Occupants", 1, 10, 3)
retrofit = st.sidebar.selectbox("🔧 Retrofit Status", [0,1], format_func=lambda x: "✅ Yes (Retrofitted)" if x==1 else "❌ No (Baseline)")

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

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
        prediction = model.predict(features_df)[0]
        
        # Display metrics in 3 columns
        st.subheader("📊 Prediction Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("⚡ Predicted Energy", f"{prediction:.2f} kWh")
        
        if retrofit == 1:
            # Kira baseline (seolah-olah tak retrofit)
            features_baseline = features_df.copy()
            features_baseline['retrofit'] = 0
            baseline_pred = model.predict(features_baseline)[0]
            
            # Kira savings
            savings = baseline_pred - prediction
            savings_pct = (savings / baseline_pred) * 100
            
            m2.metric("💰 Energy Savings", f"{savings:.2f} kWh", delta=f"{savings_pct:.1f}%")
            m3.metric("🏆 Efficiency Gain", f"{savings_pct:.1f}%", delta="Reduction")
            
            st.success(f"💡 **Retrofit Savings:** {savings:.2f} kWh ({savings_pct:.1f}% less energy)")
            
            # ==========================================
            # GRAPH 1: BAR CHART - Baseline vs Retrofitted
            # ==========================================
            st.subheader("📊 Energy Consumption Comparison")
            
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            categories = ['Baseline\n(No Retrofit)', 'Retrofitted\n(With Retrofit)']
            values = [baseline_pred, prediction]
            colors = ['#e74c3c', '#2ecc71']
            
            bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{val:.2f} kWh', ha='center', fontweight='bold', fontsize=11)
            
            # Add savings annotation
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
            # GRAPH 2: DONUT CHART - Savings Percentage
            # ==========================================
            st.subheader("📊 Savings Breakdown")
            
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            
            # Data for donut chart
            sizes = [savings, baseline_pred - savings]
            labels = [f'Savings\n({savings_pct:.1f}%)', f'Remaining Energy\n({100-savings_pct:.1f}%)']
            colors_donut = ['#2ecc71', '#e74c3c']
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_donut,
                                                autopct='%1.1f%%', startangle=90,
                                                textprops={'fontsize': 10, 'fontweight': 'bold'})
            
            # Create donut (circle in the middle)
            centre_circle = plt.Circle((0,0), 0.70, fc='white', linewidth=2)
            fig2.gca().add_artist(centre_circle)
            
            ax2.set_title('Energy Savings Distribution', fontweight='bold', fontsize=14)
            
            st.pyplot(fig2)
            
            # ==========================================
            # GRAPH 3: HORIZONTAL GAUGE - Efficiency Score
            # ==========================================
            st.subheader("📊 Efficiency Score")
            
            fig3, ax3 = plt.subplots(figsize=(8, 2))
            
            # Calculate efficiency score (higher is better - more savings)
            efficiency_score = savings_pct  # 0-100 scale
            if efficiency_score > 100:
                efficiency_score = 100
            
            # Create horizontal bar gauge
            if efficiency_score < 30:
                color = '#e74c3c'  # Red - poor
                label = 'Low Efficiency'
            elif efficiency_score < 60:
                color = '#f39c12'  # Orange - medium
                label = 'Medium Efficiency'
            else:
                color = '#2ecc71'  # Green - good
                label = 'High Efficiency'
            
            ax3.barh([0], [efficiency_score], color=color, height=0.4, edgecolor='black')
            ax3.barh([0], [100], color='lightgray', height=0.4, alpha=0.3)  # Background
            ax3.set_xlim(0, 100)
            ax3.set_yticks([])
            ax3.set_xlabel('Efficiency Score (%)', fontsize=10)
            ax3.set_title(f'Retrofit Efficiency: {label} ({efficiency_score:.1f}% savings)', fontweight='bold')
            
            # Add value label at the end of the bar
            ax3.text(efficiency_score + 2, 0, f'{efficiency_score:.1f}%', va='center', fontweight='bold')
            
            # Add color zones
            ax3.axvline(x=30, color='orange', linestyle='--', alpha=0.7, linewidth=1)
            ax3.axvline(x=60, color='green', linestyle='--', alpha=0.7, linewidth=1)
            ax3.text(15, -0.3, 'Poor', ha='center', fontsize=8, alpha=0.7)
            ax3.text(45, -0.3, 'Medium', ha='center', fontsize=8, alpha=0.7)
            ax3.text(80, -0.3, 'Good', ha='center', fontsize=8, alpha=0.7)
            
            st.pyplot(fig3)
            
            # ==========================================
            # GRAPH 4: SAVINGS BY FACTOR (if you want more analysis)
            # ==========================================
            with st.expander("📈 Advanced Analysis: Savings by Factor"):
                st.markdown("**How different factors affect your savings:**")
                
                # Create scenario variations
                scenarios = []
                savings_scenarios = []
                
                # Scenario 1: Current settings
                scenarios.append("Current")
                savings_scenarios.append(savings)
                
                # Scenario 2: Higher temperature
                features_hot = features_df.copy()
                features_hot['temperature'] = min(45, temp + 10)
                baseline_hot = model.predict(features_hot)[0]
                features_hot_retrofit = features_hot.copy()
                features_hot_retrofit['retrofit'] = 1
                retrofit_hot = model.predict(features_hot_retrofit)[0]
                savings_hot = baseline_hot - retrofit_hot
                scenarios.append(f"+10°C\n({temp+10:.0f}°C)")
                savings_scenarios.append(savings_hot)
                
                # Scenario 3: Peak hour
                features_peak = features_df.copy()
                features_peak['hour'] = 18  # 6pm peak hour
                baseline_peak = model.predict(features_peak)[0]
                features_peak_retrofit = features_peak.copy()
                features_peak_retrofit['retrofit'] = 1
                retrofit_peak = model.predict(features_peak_retrofit)[0]
                savings_peak = baseline_peak - retrofit_peak
                scenarios.append("Peak Hour\n(6pm)")
                savings_scenarios.append(savings_peak)
                
                # Scenario 4: More occupants
                features_more = features_df.copy()
                features_more['occupants'] = min(10, occupants + 3)
                baseline_more = model.predict(features_more)[0]
                features_more_retrofit = features_more.copy()
                features_more_retrofit['retrofit'] = 1
                retrofit_more = model.predict(features_more_retrofit)[0]
                savings_more = baseline_more - retrofit_more
                scenarios.append(f"+3 Occupants\n({occupants+3})")
                savings_scenarios.append(savings_more)
                
                # Create bar chart for scenarios
                fig4, ax4 = plt.subplots(figsize=(10, 5))
                bars4 = ax4.bar(scenarios, savings_scenarios, color='#3498db', edgecolor='black', linewidth=1)
                
                # Add value labels
                for bar, val in zip(bars4, savings_scenarios):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{val:.2f} kWh', ha='center', fontweight='bold', fontsize=9)
                
                ax4.set_ylabel('Potential Savings (kWh)', fontsize=12)
                ax4.set_xlabel('Scenario', fontsize=12)
                ax4.set_title('Retrofit Savings Under Different Conditions', fontweight='bold', fontsize=14)
                ax4.grid(axis='y', alpha=0.3, linestyle='--')
                
                st.pyplot(fig4)
                
                st.caption("📝 **Interpretation:** Higher savings occur in hotter conditions, peak hours, and with more occupants.")
            
            # Display comparison table
            with st.expander("📋 Detailed Comparison Table"):
                comparison_data = {
                    'Metric': ['Energy Consumption', 'Savings', 'Efficiency'],
                    'Baseline (No Retrofit)': [f"{baseline_pred:.2f} kWh", '-', '-'],
                    'Retrofitted': [f"{prediction:.2f} kWh", f"{savings:.2f} kWh", f"{savings_pct:.1f}%"]
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
        else:
            # Kalau retrofit = 0, tunjuk potential savings kalau retrofit
            features_retrofit = features_df.copy()
            features_retrofit['retrofit'] = 1
            retrofit_pred = model.predict(features_retrofit)[0]
            potential_savings = prediction - retrofit_pred
            potential_pct = (potential_savings / prediction) * 100
            
            m2.metric("💰 Potential Savings", f"{potential_savings:.2f} kWh", delta=f"{potential_pct:.1f}%")
            m3.metric("🏆 Would Save", f"{potential_pct:.1f}%", delta="If retrofitted")
            
            st.info(f"💡 **If you retrofit this building:** Would save ~{potential_savings:.2f} kWh ({potential_pct:.1f}%)")
            st.caption("👉 **Tip:** Try selecting 'Yes (Retrofitted)' above to see actual savings with detailed graphs!")
            
            # Show simple comparison graph
            st.subheader("📊 What If Analysis")
            fig_simple, ax_simple = plt.subplots(figsize=(8, 5))
            categories = ['Current\n(No Retrofit)', 'If Retrofitted']
            values = [prediction, retrofit_pred]
            colors = ['#e74c3c', '#2ecc71']
            
            bars = ax_simple.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
            for bar, val in zip(bars, values):
                ax_simple.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                              f'{val:.2f} kWh', ha='center', fontweight='bold')
            
            ax_simple.set_ylabel('Energy Consumption (kWh)', fontsize=12)
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
    """)
    
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
