from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Paths
DATA_FILE = Path("data/raw/weather_data.csv")
MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "temperature_model.joblib"

# Required columns
FEATURES = ["humidity", "pressure", "wind_speed", "rainfall"]
TARGET = "temperature"


def main():
    # 1. Check data file
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run simulator + gateway first to collect data."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Read CSV (WITH header)
    df = pd.read_csv(DATA_FILE)

    # 3. Validate columns
    expected_columns = {"timestamp", TARGET, *FEATURES, "device_id"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(
            f"CSV column mismatch.\nExpected: {expected_columns}\nFound: {df.columns.tolist()}"
        )

    # 4. Convert feature columns to numeric (extra safety)
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Drop rows with missing values
    df = df.dropna(subset=FEATURES + [TARGET])

    # 6. Ensure enough data
    if len(df) < 5:
        raise ValueError(
            f"Not enough data to train model. Collected only {len(df)} rows."
        )

    # 7. Feature matrix & target
    X = df[FEATURES]
    y = df[TARGET]

    # 8. Train / test split (safe for small datasets)
    if len(df) < 10:
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # 9. Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 10. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # 11. Save model
    joblib.dump(model, MODEL_PATH)

    # 12. Output
    print(f"Model trained and saved to: {MODEL_PATH}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
    print("ℹThis is a baseline linear regression model.")


if __name__ == "__main__":
    main()
