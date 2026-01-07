import random
import time
import json
from datetime import datetime

import paho.mqtt.client as mqtt
import sys
from pathlib import Path

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "weather/data"
SLEEP_SECONDS = 1  # send every 5 seconds


def generate_weather_point():
    """
    Generate semi-realistic hyper-local weather data.
    You can mention in report that ranges are based on typical Swedish climate (or your city).
    """
    # base realistic ranges
    temperature = round(random.uniform(-5, 35), 2)      # °C
    humidity = round(random.uniform(30, 95), 2)         # %
    pressure = round(random.uniform(980, 1035), 2)      # hPa
    wind_speed = round(random.uniform(0, 15), 2)        # m/s
    rainfall = round(max(0, random.gauss(2, 4)), 2)     # mm/hr style intensity

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "rainfall": rainfall,
        "location": "Raspberry-Pi-Lab-Spot-01"  # hyper-local point
    }


def main():
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

    print(f"Connected to MQTT at {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Publishing to topic '{MQTT_TOPIC}' every {SLEEP_SECONDS}s")

    try:
        while True:
            data = generate_weather_point()
            payload = json.dumps(data)
            client.publish(MQTT_TOPIC, payload)
            print(" Sent:", payload)
            time.sleep(SLEEP_SECONDS)
    except KeyboardInterrupt:
        print("\n Stopped simulator")


if __name__ == "__main__":
    main()
