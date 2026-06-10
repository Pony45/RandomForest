# ============================================
# EXTRA: UNIT CONVERSION & MODEL PERFORMANCE
# ============================================

# Add unit conversion selector after prediction
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Display Settings")

unit_option = st.sidebar.selectbox(
    "Display Energy Unit",
    ["Per Hour (kWh)", "Per Day (kWh)", "Per Month (kWh)", "Per Year (kWh)"],
    help="Convert prediction to different time units"
)

# Model performance metrics (hardcoded from training)
st.sidebar.markdown("---")
st.sidebar.header("📊 Model Performance")

with st.sidebar.expander("Performance Metrics", expanded=True):
    # These values are from Model 1 (Synthetic) training
    r2_score = 0.8723  # Ganti dengan R² awak
    mae = 1.234        # Ganti dengan MAE awak
    rmse = 1.567       # Ganti dengan RMSE awak
    
    st.metric("R² Score", f"{r2_score:.4f}", help="Higher is better (1.0 = perfect)")
    st.metric("MAE", f"{mae:.2f} kWh", help="Mean Absolute Error - lower is better")
    st.caption(f"RMSE: {rmse:.2f} kWh")
    st.progress(r2_score, text=f"Accuracy: {r2_score*100:.1f}%")

# Modify prediction display to show converted units
# Add this function to convert prediction
def convert_energy(prediction_kwh, target_unit):
    """Convert hourly kWh to different time units"""
    if target_unit == "Per Hour (kWh)":
        return prediction_kwh
    elif target_unit == "Per Day (kWh)":
        return prediction_kwh * 24
    elif target_unit == "Per Month (kWh)":
        return prediction_kwh * 24 * 30
    elif target_unit == "Per Year (kWh)":
        return prediction_kwh * 24 * 365

# Also add unit label
def get_unit_label(unit):
    if unit == "Per Hour (kWh)":
        return "kWh"
    elif unit == "Per Day (kWh)":
        return "kWh/day"
    elif unit == "Per Month (kWh)":
        return "kWh/month"
    elif unit == "Per Year (kWh)":
        return "kWh/year"

# Optional: Add feature importance chart (untuk tunjuk kat SV)
def plot_feature_importance():
    """Plot feature importance from the trained model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get feature names from FEATURES
        if len(importances) == len(FEATURES):
            imp_df = pd.DataFrame({
                'Feature': FEATURES,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(imp_df)))
            bars = ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors, edgecolor='black')
            
            for bar, val in zip(bars, imp_df['Importance']):
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f} ({val*100:.1f}%)', va='center', fontsize=9)
            
            ax.set_xlabel('Feature Importance Score', fontsize=12)
            ax.set_title('Random Forest - Feature Importance', fontweight='bold', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            
            return fig
    return None

# Add Feature Importance tab
tab4 = st.tabs(["📊 Feature Importance"])[0] if len(st.tabs(["dummy"])) > 3 else None

# Better approach: Add as new section
st.markdown("---")
st.header("📊 Model Feature Importance")

if st.button("Show Feature Importance Chart"):
    fig_imp = plot_feature_importance()
    if fig_imp:
        st.pyplot(fig_imp)
    else:
        st.info("Feature importance not available from loaded model")

# Modify the prediction display in tab1 to use unit conversion
# Find the prediction display section and modify it
