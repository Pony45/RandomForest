# ============================================
# M&V ENERGY SAVINGS DASHBOARD
# Streamlit Application for Thesis
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="M&V Energy Savings Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Predict energy savings from residential building retrofits using Random Forest*")

# ============================================
# LOAD TRAINED MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load trained Random Forest model and features"""
    model = joblib.load('models/thesis_mv_random_forest.pkl')
    
    # Load feature names
    with open('models/thesis_mv_features.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    return model, features

try:
    model, FEATURES = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except:
    st.sidebar.error("❌ Model not found. Please train model first.")
    st.stop()

# ============================================
# SIDEBAR - INPUT PARAMETERS
# ============================================
st.sidebar.header("📋 Building Parameters")

# Create two columns in sidebar
col1, col2 = st.sidebar.columns(2)

with col1:
    floor_area = st.number_input("Floor Area (m²)", min_value=30, max_value=300, value=90, step=10)
    occupants = st.number_input("Number of Occupants", min_value=1, max_value=10, value=3, step=1)
    temperature = st.slider("Temperature (°C)", min_value=-5, max_value=45, value=22.0, step=0.5)

with col2:
    humidity = st.slider("Humidity (%)", min_value=20, max_value=100, value=60, step=5)
    retrofit = st.selectbox("Retrofit Status", ["No (Baseline)", "Yes (Retrofitted)"])
    hour = st.slider("Hour of Day", min_value=0, max_value=23, value=14, step=1)

# Date and time inputs
date = st.sidebar.date_input("Date", datetime.now())
dayofweek = date.weekday()  # 0=Monday, 6=Sunday
month = date.month

# Additional features
is_weekend = 1 if dayofweek >= 5 else 0

# ============================================
# FEATURE ENGINEERING (same as training)
# ============================================
def engineer_features(hour, month, temperature, humidity, floor_area, occupants, retrofit):
    """Apply same feature engineering as training"""
    retrofit_value = 1 if retrofit == "Yes (Retrofitted)" else 0
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Interaction features
    temp_humidity_interaction = temperature * humidity / 100
    occupants_per_area = occupants / floor_area
    
    features_dict = {
        'temperature': temperature,
        'humidity': humidity,
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'floor_area': floor_area,
        'occupants': occupants,
        'retrofit': retrofit_value,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'is_weekend': is_weekend,
        'temp_humidity_interaction': temp_humidity_interaction,
        'occupants_per_area': occupants_per_area
    }
    
    # Create DataFrame with correct feature order
    feature_df = pd.DataFrame([features_dict])[FEATURES]
    
    return feature_df, retrofit_value

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_energy(features_df):
    """Make prediction using trained model"""
    prediction = model.predict(features_df)[0]
    return prediction

# ============================================
# MAIN CONTENT - THREE SCENARIOS
# ============================================
st.header("🎯 Energy Consumption Prediction")

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["📊 Single Prediction", "📈 Savings Analysis", "🔍 Feature Impact"])

# ===== TAB 1: Single Prediction =====
with tab1:
    st.subheader("Instant Energy Prediction")
    
    # Prepare features
    features_df, retrofit_value = engineer_features(
        hour, month, temperature, humidity, 
        floor_area, occupants, retrofit
    )
    
    # Make prediction
    if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
        prediction = predict_energy(features_df)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="⚡ Predicted Energy Consumption",
                value=f"{prediction:.2f} kWh",
                delta=None
            )
        
        with col2:
            # Compare with typical
            typical = 12.5  # You can calculate this from your data
            delta = prediction - typical
            st.metric(
                label="Compared to Typical Building",
                value=f"{prediction:.2f} kWh",
                delta=f"{delta:+.2f} kWh",
                delta_color="inverse"
            )
        
        with col3:
            retrofit_status = "✅ Retrofitted" if retrofit_value == 1 else "❌ Not Retrofitted"
            st.metric(
                label="Building Status",
                value=retrofit_status,
                delta=None
            )
        
        # Gauge chart
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Create gauge
        gauge_colors = ['#2ecc71', '#f39c12', '#e74c3c']
        if prediction <= 10:
            color = gauge_colors[0]
            label = "Excellent (Low Energy)"
        elif prediction <= 15:
            color = gauge_colors[1]
            label = "Average"
        else:
            color = gauge_colors[2]
            label = "High (Optimization Needed)"
        
        ax.barh([0], [prediction], color=color, height=0.5)
        ax.set_xlim(0, 25)
        ax.set_yticks([])
        ax.set_xlabel("Energy Consumption (kWh)")
        ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=15, color='gray', linestyle='--', alpha=0.5)
        ax.text(prediction + 0.5, 0, f"{prediction:.1f} kWh", va='center')
        ax.set_title(f"Energy Efficiency Rating: {label}")
        
        st.pyplot(fig)

# ===== TAB 2: Savings Analysis =====
with tab2:
    st.subheader("💰 Energy Savings from Retrofit")
    
    # Calculate baseline (no retrofit) vs retrofit scenario
    features_baseline, _ = engineer_features(
        hour, month, temperature, humidity,
        floor_area, occupants, "No (Baseline)"
    )
    
    features_retrofit, _ = engineer_features(
        hour, month, temperature, humidity,
        floor_area, occupants, "Yes (Retrofitted)"
    )
    
    pred_baseline = predict_energy(features_baseline)
    pred_retrofit = predict_energy(features_retrofit)
    
    savings = pred_baseline - pred_retrofit
    savings_pct = (savings / pred_baseline) * 100
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🏚️ Baseline (No Retrofit)",
            value=f"{pred_baseline:.2f} kWh",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🏠 Retrofitted",
            value=f"{pred_retrofit:.2f} kWh",
            delta=f"-{savings:.2f} kWh",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="💵 Energy Savings",
            value=f"{savings:.2f} kWh",
            delta=f"{savings_pct:.1f}% reduction",
            delta_color="normal"
        )
    
    # Financial savings (assuming electricity price)
    electricity_price = st.number_input("Electricity Price (RM/kWh)", min_value=0.10, max_value=1.50, value=0.52, step=0.01)
    daily_savings_rm = savings * electricity_price
    yearly_savings_rm = daily_savings_rm * 365
    
    st.info(f"""
    💰 **Financial Impact (at RM {electricity_price}/kWh):**
    - Daily savings: **RM {daily_savings_rm:.2f}**
    - Yearly savings: **RM {yearly_savings_rm:.2f}**
    - 10-year savings: **RM {yearly_savings_rm * 10:,.2f}**
    """)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['Baseline\n(No Retrofit)', 'Retrofitted'], 
                  [pred_baseline, pred_retrofit],
                  color=['#e74c3c', '#2ecc71'],
                  edgecolor='black',
                  linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, [pred_baseline, pred_retrofit]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{value:.2f} kWh', ha='center', fontweight='bold')
    
    # Add savings annotation
    ax.annotate(f'💡 Savings: {savings:.2f} kWh\n({savings_pct:.1f}%)',
                xy=(1, pred_retrofit + savings/2),
                xytext=(1.5, pred_retrofit + savings/2 + 1),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Energy Consumption (kWh)')
    ax.set_title('Retrofit Impact Analysis', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    st.pyplot(fig)

# ===== TAB 3: Feature Impact Analysis =====
with tab3:
    st.subheader("🔍 How Each Feature Affects Energy Consumption")
    
    # Feature to analyze
    feature_to_vary = st.selectbox(
        "Select feature to analyze:",
        ['temperature', 'humidity', 'hour', 'occupants', 'floor_area']
    )
    
    # Create range of values
    if feature_to_vary == 'temperature':
        x_range = np.arange(-5, 46, 1)
        x_label = "Temperature (°C)"
        fixed_values = {'temperature': 22, 'humidity': 60, 'hour': 14, 
                       'occupants': 3, 'floor_area': 90}
    elif feature_to_vary == 'humidity':
        x_range = np.arange(20, 101, 5)
        x_label = "Humidity (%)"
        fixed_values = {'temperature': 22, 'humidity': 60, 'hour': 14,
                       'occupants': 3, 'floor_area': 90}
    elif feature_to_vary == 'hour':
        x_range = np.arange(0, 24, 1)
        x_label = "Hour of Day"
        fixed_values = {'temperature': 22, 'humidity': 60, 'hour': 14,
                       'occupants': 3, 'floor_area': 90}
    elif feature_to_vary == 'occupants':
        x_range = np.arange(1, 11, 1)
        x_label = "Number of Occupants"
        fixed_values = {'temperature': 22, 'humidity': 60, 'hour': 14,
                       'occupants': 3, 'floor_area': 90}
    else:  # floor_area
        x_range = np.arange(30, 301, 10)
        x_label = "Floor Area (m²)"
        fixed_values = {'temperature': 22, 'humidity': 60, 'hour': 14,
                       'occupants': 3, 'floor_area': 90}
    
    # Predict for baseline and retrofit scenarios
    predictions_baseline = []
    predictions_retrofit = []
    
    for x_val in x_range:
        # Baseline (no retrofit)
        fixed_values[feature_to_vary] = x_val
        features_baseline, _ = engineer_features(
            fixed_values['hour'], 14,  # month fixed to July
            fixed_values['temperature'],
            fixed_values['humidity'],
            fixed_values['floor_area'],
            fixed_values['occupants'],
            "No (Baseline)"
        )
        predictions_baseline.append(predict_energy(features_baseline))
        
        # Retrofit
        features_retrofit, _ = engineer_features(
            fixed_values['hour'], 14,
            fixed_values['temperature'],
            fixed_values['humidity'],
            fixed_values['floor_area'],
            fixed_values['occupants'],
            "Yes (Retrofitted)"
        )
        predictions_retrofit.append(predict_energy(features_retrofit))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_range, predictions_baseline, 'o-', label='Baseline (No Retrofit)', 
            color='#e74c3c', linewidth=2, markersize=6)
    ax.plot(x_range, predictions_retrofit, 's-', label='Retrofitted', 
            color='#2ecc71', linewidth=2, markersize=6)
    
    # Fill between to show savings
    ax.fill_between(x_range, predictions_baseline, predictions_retrofit, 
                    alpha=0.3, color='blue', label='Potential Savings')
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax.set_title(f'Impact of {feature_to_vary.replace("_", " ").title()} on Energy Consumption', 
                 fontweight='bold', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Add interpretation
    st.markdown("""
    ### 📝 Interpretation:
    - **Red line (Baseline)**: Energy consumption if building is NOT retrofitted
    - **Green line (Retrofitted)**: Energy consumption after retrofit
    - **Blue shaded area**: Energy savings achievable through retrofit
    - **Larger gap** = Retrofit is more effective under these conditions
    """)

# ============================================
# SIDEBAR - BATCH PREDICTION (Optional)
# ============================================
st.sidebar.markdown("---")
st.sidebar.header("📁 Batch Prediction")

uploaded_file = st.sidebar.file_uploader("Upload CSV for batch prediction", type=['csv'])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ Loaded {len(batch_df)} records")
    
    if st.sidebar.button("Run Batch Prediction"):
        # Process batch predictions
        predictions = []
        for idx, row in batch_df.iterrows():
            features_df, _ = engineer_features(
                row['hour'], row['month'], row['temperature'],
                row['humidity'], row['floor_area'], row['occupants'],
                "Yes (Retrofitted)" if row.get('retrofit', 0) == 1 else "No (Baseline)"
            )
            pred = predict_energy(features_df)
            predictions.append(pred)
        
        batch_df['predicted_energy'] = predictions
        st.sidebar.download_button(
            label="📥 Download Predictions",
            data=batch_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
        st.sidebar.success("✅ Predictions ready for download!")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎓 AI-based Measurement & Verification (M&V) System | Random Forest Model | Thesis Project</p>
    <p>⚠️ This is a demonstration tool. Actual energy savings should be verified with utility bills.</p>
</div>
""", unsafe_allow_html=True)