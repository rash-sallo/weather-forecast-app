import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="Weather Forecasting System",
    page_icon="🌤️",
    layout="wide"
)

# ── Custom Styling ────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #90caf9;
    }
    .metric-label {
        font-size: 13px;
        color: #aaaaaa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    section[data-testid="stSidebar"] {
        background-color: #16213e;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model & Scalers ──────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'weather_cnn_model.h5'), compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    scaler_X = joblib.load(os.path.join(BASE_DIR, 'scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(BASE_DIR, 'scaler_y.pkl'))
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model()

# ── Sidebar Navigation ────────────────────────────────
st.sidebar.title("🌤️ Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Predict Temperature", "About the Model"])

st.sidebar.divider()
st.sidebar.caption("Final Year Project")
st.sidebar.caption("Computer Science Department")

# ══════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════
if page == "Overview":
    st.title("🌤️ Weather Forecasting System")
    st.markdown("**Mutual Information feature selection + Convolutional Neural Network**")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4 / 5</div>
            <div class="metric-label">Features Used</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">CNN</div>
            <div class="metric-label">Model Type</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">24 hrs</div>
            <div class="metric-label">Sequence Window</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Shannon MI</div>
            <div class="metric-label">Selection Method</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("📊 Selected Features (by Mutual Information score)")
    feat_df = pd.DataFrame({
        "Rank": [1, 2, 3, 4],
        "Feature": ["Dew Point Temp (°C)", "Pressure (kPa)", "Relative Humidity (%)", "Visibility (km)"],
        "Status": ["✅ Selected", "✅ Selected", "✅ Selected", "✅ Selected"]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
    st.caption("Wind Speed was dropped — its Mutual Information score fell below the 0.1 threshold.")

    st.divider()
    st.info("👈 Use the sidebar to try a live prediction or read about the model.")

# ══════════════════════════════════════════════════════
# PAGE 2: PREDICT
# ══════════════════════════════════════════════════════
elif page == "Predict Temperature":
    st.title("🔍 Predict Temperature")
    st.markdown("Enter current weather conditions to forecast temperature.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        dew_point = st.slider("Dew Point Temp (°C)", -30.0, 25.0, 3.0)
        humidity  = st.slider("Relative Humidity (%)", 0, 100, 70)
    with col2:
        pressure   = st.slider("Pressure (kPa)", 97.0, 104.0, 101.0)
        visibility = st.slider("Visibility (km)", 0.0, 50.0, 25.0)

    if 'history' not in st.session_state:
        st.session_state.history = []

    if st.button("🔍 Predict Temperature", use_container_width=True, type="primary"):
        features = np.array([[dew_point, pressure, humidity, visibility]])
        features_scaled = scaler_X.transform(features)
        features_seq = np.repeat(features_scaled, 24, axis=0)
        features_reshaped = features_seq.reshape(1, 24, features_scaled.shape[1])

        pred_scaled = model.predict(features_reshaped, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)
        temp = round(float(pred[0][0]), 2)

        if temp < 0:
            condition, emoji = "Freezing Cold", "🥶"
        elif temp < 10:
            condition, emoji = "Cold", "🌨️"
        elif temp < 20:
            condition, emoji = "Mild", "🌤️"
        elif temp < 30:
            condition, emoji = "Warm", "☀️"
        else:
            condition, emoji = "Hot", "🔥"

        st.divider()
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.metric("🌡️ Forecasted Temperature", f"{temp} °C")
        with rcol2:
            st.metric("Condition", f"{emoji} {condition}")

        st.session_state.history.append({
            'Dew Point': dew_point, 'Pressure': pressure,
            'Humidity': humidity, 'Visibility': visibility,
            'Predicted Temp (°C)': temp
        })

    if st.session_state.history:
        st.divider()
        st.subheader("📋 Prediction History")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.line_chart(df['Predicted Temp (°C)'])

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ══════════════════════════════════════════════════════
else:
    st.title("📖 About the Model")
    st.divider()

    st.markdown("""
    ### Pipeline

    1. **Data** — Hourly weather records (Temperature, Dew Point, Humidity, Wind Speed, Visibility, Pressure)
    2. **Feature Selection** — Mutual Information scores each feature's statistical dependency on temperature; features above a 0.1 threshold are kept
    3. **Sequencing** — The last 24 hourly readings are stacked into one input window
    4. **Model** — A 1D Convolutional Neural Network learns patterns across the time window to forecast the next hour's temperature
    5. **Evaluation** — Performance measured with MSE, MAE, and RMSE on a held-out test set
    """)

    st.divider()
    st.subheader("⚠️ Limitation")
    st.warning(
        "This live demo repeats your single input across the 24-hour window for simplicity. "
        "In a full deployment, the model would use 24 actual historical hourly readings for a more accurate forecast."
    )
