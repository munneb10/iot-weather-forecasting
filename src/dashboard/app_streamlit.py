from pathlib import Path
import time
import pandas as pd
import streamlit as st
import subprocess
import sys
import json
import paho.mqtt.client as mqtt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from src.ml.model_predict import TemperaturePredictor

DATA_FILE = Path("data/raw/weather_data.csv")

# GLOBAL THREAD-SAFE MQTT BUFFER
mqtt_buffer = []

# ============================
# LOAD CSV DATA
# ============================
def load_data():
    if not DATA_FILE.exists():
        return pd.DataFrame(columns=[
            "timestamp", "temperature", "humidity",
            "pressure", "wind_speed", "rainfall", "location"
        ])

    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

def main():
    st.set_page_config(page_title="IoT Weather Dashboard", layout="wide")
    st.title("Hyper-Local IoT Weather Forecasting System")

    # ============================================================
    # STATIC UI
    # ============================================================
    st.subheader("Model Control")

    if st.button("Train on new data", key="train_button"):
        with st.spinner("Training model..."):
            result = subprocess.run(
                [sys.executable, "src/ml/train_model.py"],
                capture_output=True,
                text=True
            )
        st.success("Model retrained successfully!")
        st.text(result.stdout)
        st.rerun()

    st.markdown("---")

    # Define Placeholders in Order of Appearance
    metrics_placeholder = st.empty()
    charts_placeholder = st.empty()
    forecast_placeholder = st.empty()  # NEW: Dedicated slot for the comparison graph
    prediction_placeholder = st.empty()
    mqtt_placeholder = st.empty()

    st.markdown("---")
    st.subheader("What-if Scenario (interactive, static)")

    colA, colB = st.columns(2)
    colC, colD = st.columns(2)

    humidity_slider = colA.slider("Humidity (%)", 0, 100, 50, key="humidity_slider")
    pressure_slider = colB.slider("Pressure (hPa)", 950, 1050, 1000, key="pressure_slider")
    wind_slider = colC.slider("Wind speed (m/s)", 0, 25, 5, key="wind_slider")
    rain_slider = colD.slider("Rainfall (mm)", 0, 50, 10, key="rain_slider")

    whatif_output = st.empty()

    # ============================================================
    # MQTT SUBSCRIBER
    # ============================================================
    MQTT_BROKER = "localhost"
    MQTT_PORT = 1883
    MQTT_TOPIC = "weather/data"

    def on_mqtt_message(client, userdata, msg):
        global mqtt_buffer
        try:
            payload = json.loads(msg.payload.decode())
        except:
            payload = msg.payload.decode()
        mqtt_buffer.append(payload)
        mqtt_buffer = mqtt_buffer[-5:]

    mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_message = on_mqtt_message
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.subscribe(MQTT_TOPIC)
        mqtt_client.loop_start()
    except Exception as e:
        st.error(f"MQTT Connection Failed: {e}")

    # ============================================================
    # REALTIME LOOP
    # ============================================================
    while True:
        df = load_data()
        
        # Initialize predictor once per refresh
        try:
            predictor = TemperaturePredictor()
        except:
            predictor = None

        if df.empty:
            metrics_placeholder.warning("Waiting for data...")
            time.sleep(1)
            continue

        df = df.sort_values("timestamp")
        latest = df.iloc[-1]

        # 1. METRICS
        with metrics_placeholder.container():
            col2, col3 = st.columns(2)
            col2.metric("Humidity (%)", f"{latest['humidity']:.2f}")
            col3.metric("Pressure (hPa)", f"{latest['pressure']:.2f}")
            col4, col5 = st.columns(2)
            col4.metric("Wind Speed (m/s)", f"{latest['wind_speed']:.2f}")
            col5.metric("Rainfall (mm)", f"{latest['rainfall']:.2f}")

        # 2. HISTORICAL TREND CHARTS
        with charts_placeholder.container():
            st.subheader("Historical Trends")
            tabs = st.tabs(["Humidity", "Pressure", "Wind + Rainfall"])
            with tabs[0]:
                st.line_chart(df.set_index("timestamp")["humidity"])
            with tabs[1]:
                st.line_chart(df.set_index("timestamp")["pressure"])
            with tabs[2]:
                st.line_chart(df.set_index("timestamp")[["wind_speed", "rainfall"]])

        # 3. TEMPERATURE FORECAST (ACTUAL VS PREDICTED)
        with forecast_placeholder.container():
            st.subheader("Model Performance: Actual vs. Predicted Temperature")
            if predictor:
                try:
                    # Use last 50 samples to keep UI fast
                    plot_df = df.tail(50).copy()
                    plot_df['predicted_temp'] = plot_df.apply(
                        lambda row: predictor.predict_temp(
                            humidity=row["humidity"],
                            pressure=row["pressure"],
                            wind_speed=row["wind_speed"],
                            rainfall=row["rainfall"]
                        ), axis=1
                    )
                    chart_data = plot_df.set_index("timestamp")[["temperature", "predicted_temp"]]
                    chart_data.columns = ["Actual Temperature", "Model Prediction"]
                    st.line_chart(chart_data)
                except Exception as e:
                    st.error(f"Graph error: {e}")

        # 4. SINGLE MODEL PREDICTION METRIC
        with prediction_placeholder.container():
            st.subheader("Live Forecast Analysis")
            if predictor:
                try:
                    pred_temp = predictor.predict_temp(
                        humidity=latest["humidity"],
                        pressure=latest["pressure"],
                        wind_speed=latest["wind_speed"],
                        rainfall=latest["rainfall"],
                    )
                    st.metric(
                        "Predicted Current Temp (°C)",
                        f"{pred_temp:.2f}",
                        delta=f"{pred_temp - latest['temperature']:.2f}"
                    )
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        # 5. MQTT LIVE VIEWER
        with mqtt_placeholder.container():
            st.subheader("Live MQTT Sensor Messages (latest 5)")
            st.session_state.mqtt_messages = list(mqtt_buffer)
            if not st.session_state.mqtt_messages:
                st.info("Waiting for MQTT messages...")
            else:
                for msg in reversed(st.session_state.mqtt_messages):
                    st.json(msg)

        # 6. WHAT-IF PREDICTION (Update the static output at the bottom)
        if predictor:
            try:
                whatif_prediction = predictor.predict_temp(
                    humidity_slider,
                    pressure_slider,
                    wind_slider,
                    rain_slider,
                )
                with whatif_output.container():
                    st.metric("What-if Predicted Temperature", f"{whatif_prediction:.2f} °C")
            except:
                with whatif_output.container():
                    st.error("Model not available for What-if")

        time.sleep(2)

if __name__ == "__main__":
    main()