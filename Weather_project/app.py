import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import tensorflow as tf

st.set_page_config(
    page_title="Weather Forecasting App",
    page_icon="🌤️",
    layout="wide"
)

st.sidebar.title("🌦️ Weather Forecasting")
page = st.sidebar.radio(
    "Select Model",
    ["📊 Mutual Information (Feature Selection)", "🧠 CNN Weather Forecasting"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MUTUAL INFORMATION
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Mutual Information (Feature Selection)":

    st.title("📊 Mutual Information – Feature Selection")
    st.markdown(
        "Mutual Information (MI) measures the dependency between each "
        "feature and the target variable (Temperature). Higher MI score = "
        "more relevant feature."
    )

    mi_data = {
        "Feature": [
            "Dew Point Temp_C",
            "Press_kPa",
            "Rel Hum_%",
            "Visibility_km",
            "Wind Speed_km/h"
        ],
        "MI Score": [1.249577, 0.225829, 0.225591, 0.122104, 0.038495],
        "Selected": [True, True, True, True, False]
    }
    df_mi = pd.DataFrame(mi_data)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("MI Score Bar Chart")
        colors = ["#1f77b4" if sel else "#d62728" for sel in df_mi["Selected"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(df_mi["Feature"], df_mi["MI Score"], color=colors)
        ax.set_title("Mutual Information Scores", fontsize=13, fontweight="bold")
        ax.set_ylabel("MI Score")
        ax.set_xlabel("Features")
        plt.xticks(rotation=20, ha="right")
        for bar, score in zip(bars, df_mi["MI Score"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{score:.4f}", ha="center", va="bottom", fontsize=9)
        legend_elements = [
            Patch(facecolor="#1f77b4", label="Selected"),
            Patch(facecolor="#d62728", label="Not Selected")
        ]
        ax.legend(handles=legend_elements)
        st.pyplot(fig)

    with col2:
        st.subheader("Feature Rankings")
        df_display = df_mi.copy()
        df_display["Selected"] = df_display["Selected"].map(
            {True: "Yes", False: "No"}
        )
        st.dataframe(df_display, use_container_width=True, height=230)
        st.success(
            "**Selected Features:**\n"
            "- Dew Point Temp_C\n"
            "- Press_kPa\n"
            "- Rel Hum_%\n"
            "- Visibility_km"
        )
        st.info("**Excluded:** Wind Speed_km/h (MI score: 0.0385)")

    st.markdown("---")
    st.subheader("How Mutual Information Works")
    st.markdown("""
    Mutual Information quantifies how much knowing one variable reduces
    uncertainty about another:

    > **I(X; Y) = H(Y) - H(Y | X)**

    - A score of **0** means the feature is independent of the target.
    - A **higher score** means the feature is more informative.

    **Dew Point Temp_C** has the highest MI score (1.2496), making it
    the most significant predictor of temperature in this dataset.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CNN FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🧠 CNN Weather Forecasting Model")
    st.markdown(
        "Enter the 4 MI-selected features below. The trained 1D CNN model "
        "will predict the **temperature (°C)**."
    )

    @st.cache_resource
    def load_model():
        model_path = os.path.join(os.path.dirname(__file__), "weather_cnn_model.h5")
        return tf.keras.models.load_model(model_path)

    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.warning(f"Model file not found or failed to load: {e}")

    st.subheader("Input Features")
    col1, col2 = st.columns(2)

    with col1:
        dew_point = st.number_input(
            "Dew Point Temp (°C)",
            min_value=-50.0, max_value=50.0, value=10.0, step=0.1,
            help="Highest MI score — most influential feature"
        )
        pressure = st.number_input(
            "Pressure (kPa)",
            min_value=90.0, max_value=110.0, value=101.3, step=0.1
        )

    with col2:
        humidity = st.number_input(
            "Relative Humidity (%)",
            min_value=0.0, max_value=100.0, value=65.0, step=0.1
        )
        visibility = st.number_input(
            "Visibility (km)",
            min_value=0.0, max_value=50.0, value=20.0, step=0.1
        )

    st.markdown("---")

    if st.button("🔍 Predict Temperature", use_container_width=True):
        if not model_loaded:
            st.error("Cannot predict: model file not found.")
        else:
            input_data = np.array([[dew_point, pressure, humidity, visibility]])
            input_data = input_data.reshape((1, 4, 1))
            prediction = model.predict(input_data)
            predicted_temp = float(prediction[0][0])
            st.success(f"### 🌡️ Predicted Temperature: **{predicted_temp:.2f} °C**")

            summary_df = pd.DataFrame({
                "Feature": ["Dew Point Temp_C", "Press_kPa", "Rel Hum_%", "Visibility_km"],
                "Value": [dew_point, pressure, humidity, visibility],
                "MI Score": [1.249577, 0.225829, 0.225591, 0.122104]
            })
            st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Architecture")
    st.markdown("""
    | Layer | Details |
    |---|---|
    | Input | Shape (4, 1) — 4 MI-selected features |
    | Conv1D | 64 filters, kernel size 2, ReLU |
    | MaxPooling1D | Pool size 2 |
    | Flatten | — |
    | Dense | 50 units, ReLU |
    | Output | 1 unit (Temperature prediction) |
    """)
