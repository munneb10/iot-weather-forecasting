from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("data/models/temperature_model.joblib")


class TemperaturePredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Train it with train_model.py first."
            )
        self.model = joblib.load(model_path)

    def predict_temp(self, humidity, pressure, wind_speed, rainfall):
        df = pd.DataFrame(
            [[humidity, pressure, wind_speed, rainfall]],
            columns=["humidity", "pressure", "wind_speed", "rainfall"],
        )
        pred = self.model.predict(df)[0]
        return float(pred)
