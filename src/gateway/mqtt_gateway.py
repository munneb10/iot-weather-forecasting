import os
import json
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt
import pandas as pd

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "weather/data"

DATA_DIR = Path("data/raw")
DATA_FILE = DATA_DIR / "weather_data.csv"


def ensure_data_file():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "temperature",
                "humidity",
                "pressure",
                "wind_speed",
                "rainfall",
                "location",
            ]
        )
        df.to_csv(DATA_FILE, index=False)


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Gateway connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        print("Failed to connect, return code:", rc)


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)

        # Ensure all expected keys exist
        record = {
            "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
            "temperature": data.get("temperature"),
            "humidity": data.get("humidity"),
            "pressure": data.get("pressure"),
            "wind_speed": data.get("wind_speed"),
            "rainfall": data.get("rainfall"),
            "location": data.get("location", "unknown"),
        }

        df = pd.DataFrame([record])
        df.to_csv(DATA_FILE, mode="a", header=not DATA_FILE.exists(), index=False)

        print("💾 Stored:", record)
    except Exception as e:
        print("⚠️ Error processing message:", e)


def main():
    ensure_data_file()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    print("IoT Gateway started, waiting for messages...")

    client.loop_forever()


if __name__ == "__main__":
    main()
