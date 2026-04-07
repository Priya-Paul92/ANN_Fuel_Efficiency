import streamlit as st
import numpy as np
import joblib
import json
import tf_keras as keras
 
# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fuel Efficiency Predictor",
    page_icon="🚗",
    layout="centered"
)
 
# ── Load Model & Scaler ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = keras.models.load_model('best_model.keras')
    scaler = joblib.load('scaler.pkl')
    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns
 
model, scaler, feature_columns = load_artifacts()
 
# ── Header ─────────────────────────────────────────────────────────────────
st.title("🚗 Fuel Efficiency Predictor")
st.markdown("### Predict Miles Per Gallon (MPG) using an Artificial Neural Network")
st.markdown("---")
 
# ── Sidebar Info ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")
    st.success("Model: ANN (Keras Tuner optimized)")
    st.info("R² Score : 0.917")
    st.info("MAE      : 1.68 MPG")
    st.info("RMSE     : 2.11 MPG")
    st.markdown("---")
    st.markdown("**Features used:**")
    for col in feature_columns:
        st.markdown(f"- {col}")
 
# ── Input Form ─────────────────────────────────────────────────────────────
st.subheader("🔧 Enter Car Specifications")
 
col1, col2 = st.columns(2)
 
with col1:
    cylinders    = st.selectbox("Cylinders", options=[3, 4, 5, 6, 8], index=1)
    displacement = st.number_input("Displacement (cu. inches)", min_value=50.0,  max_value=500.0, value=150.0, step=1.0)
    horsepower   = st.number_input("Horsepower",                min_value=40.0,  max_value=250.0, value=100.0, step=1.0)
    weight       = st.number_input("Weight (lbs)",              min_value=1500.0, max_value=5500.0, value=2500.0, step=10.0)
 
with col2:
    acceleration = st.number_input("Acceleration (0-60 mph, sec)", min_value=8.0,  max_value=25.0, value=15.0, step=0.1)
    model_year   = st.slider("Model Year", min_value=70, max_value=82, value=76)
    origin       = st.radio("Origin", options=["USA", "Europe", "Japan"])
 
st.markdown("---")
 
# ── Predict Button ─────────────────────────────────────────────────────────
if st.button("🚀 Predict MPG", use_container_width=True):
 
    # One-hot encode origin
    usa    = 1.0 if origin == "USA"    else 0.0
    europe = 1.0 if origin == "Europe" else 0.0
    japan  = 1.0 if origin == "Japan"  else 0.0
 
    # Build input in correct feature order
    input_data = np.array([[
        cylinders,
        displacement,
        horsepower,
        weight,
        acceleration,
        model_year,
        usa,
        europe,
        japan
    ]])
 
    # Scale input
    input_scaled = scaler.transform(input_data)
 
    # Predict
    prediction = model.predict(input_scaled, verbose=0)[0][0]
 
    # ── Result Display ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
 
    col_r1, col_r2, col_r3 = st.columns(3)
 
    with col_r1:
        st.metric(label="Predicted MPG", value=f"{prediction:.2f}")
 
    with col_r2:
        # Fuel efficiency rating
        if prediction >= 30:
            rating = "🟢 Excellent"
        elif prediction >= 20:
            rating = "🟡 Average"
        else:
            rating = "🔴 Poor"
        st.metric(label="Efficiency Rating", value=rating)
 
    with col_r3:
        # Litres per 100km conversion
        lper100 = 235.214 / prediction
        st.metric(label="L/100km", value=f"{lper100:.2f}")
 
    st.success(f"✅ This car is predicted to achieve **{prediction:.2f} MPG** fuel efficiency!")
 
    # ── Input Summary ──────────────────────────────────────────────────────
    with st.expander("📋 View Input Summary"):
        import pandas as pd
        summary = pd.DataFrame({
            'Feature': ['Cylinders', 'Displacement', 'Horsepower', 'Weight',
                        'Acceleration', 'Model Year', 'Origin'],
            'Value'  : [cylinders, displacement, horsepower, weight,
                        acceleration, model_year, origin]
        })
        st.dataframe(summary, use_container_width=True)
 
# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with TensorFlow, Keras Tuner & Streamlit</p>",
    unsafe_allow_html=True
)
