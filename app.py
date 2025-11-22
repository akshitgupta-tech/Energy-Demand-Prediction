from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import joblib
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load Models ----------
xgb_model = pickle.load(open("models/vmd_xgb_model.pkl", "rb"))
lstm_model = tf.keras.models.load_model("models/vmd_lstm_model.h5")
feature_scaler = pickle.load(open("models/feature_scaler.pkl", "rb"))
target_scaler = pickle.load(open("models/target_scaler.pkl", "rb"))
blend_weight = pickle.load(open("models/blend_weight.pkl", "rb"))  # value 0-1

# ---------- Prediction API ----------
@app.post("/predict")
def predict(payload: dict):
    hour = payload["hour"]
    region = payload["region"]

    # Create 30 synthetic feature rows for next 30 days
    X = np.array([[hour]] * 30)

    # Scale input
    X_scaled = feature_scaler.transform(X)

    # Model predictions
    pred_xgb = xgb_model.predict(X_scaled)
    pred_lstm = lstm_model.predict(X_scaled).flatten()

    # Blend
    w = blend_weight
    blended = (w * pred_lstm) + ((1 - w) * pred_xgb)

    # Inverse scale
    final_pred = target_scaler.inverse_transform(blended.reshape(-1, 1)).flatten()

    # Round + output
    output = []
    for i, val in enumerate(final_pred):
        output.append({
            "day": i + 1,
            "demand": max(0, round(float(val)))
        })

    return {"region": region, "forecast": output}
