import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def predict_demand(hour, horizon_days):
    # Load scalers, blend weight, models
    scaler_X = joblib.load("models/feature_scaler.pkl")
    scaler_y = joblib.load("models/target_scaler.pkl")
    blend_weight = joblib.load("models/blend_weight.pkl")
    xgb_full = joblib.load("models/vmd_xgb_model.pkl")
    lstm_full = tf.keras.models.load_model("models/vmd_lstm_model.h5")
    SEQ_LEN = 168

    # Load data and build features (same as your training)
    df = pd.read_excel("hourlyLoadDataIndia.xlsx")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    df["Load"] = df["National Hourly Demand"] if "National Hourly Demand" in df.columns else df["Load"]

    # Feature engineering
    df["lag_1"] = df["Load"].shift(1)
    df["lag_24"] = df["Load"].shift(24)
    df["lag_168"] = df["Load"].shift(168)
    df["lag_336"] = df["Load"].shift(336)
    df["rm_24"] = df["Load"].rolling(24).mean()
    df["rm_168"] = df["Load"].rolling(168).mean()
    df["sin_hour"] = np.sin(2*np.pi*df.index.hour/24)
    df["cos_hour"] = np.cos(2*np.pi*df.index.hour/24)
    df["sin_dow"]  = np.sin(2*np.pi*df.index.dayofweek/7)
    df["cos_dow"]  = np.cos(2*np.pi*df.index.dayofweek/7)

    feature_cols = [
        "lag_1","lag_24","lag_168","lag_336",
        "rm_24","rm_168",
        "sin_hour","cos_hour","sin_dow","cos_dow"
    ]
    # Add existing IMFs if any (from training, otherwise drop)
    imf_cols = [col for col in df.columns if col.startswith("IMF_")]
    feature_cols += imf_cols

    df = df.dropna()
    # Filter for selected hour
    df = df[df.index.strftime('%H:%M') == hour]
    df = df.tail(1000)  # Use recent data for robustness

    X_all = scaler_X.transform(df[feature_cols])
    # Get last known sequence for prediction
    X_seq = X_all[-SEQ_LEN:].reshape((1, SEQ_LEN, len(feature_cols)))
    X_xgb = X_seq.reshape((1, -1))

    # Get base predictions
    pred_xgb = xgb_full.predict(X_xgb)
    pred_lstm = lstm_full.predict(X_seq)[0]

    # Hybrid blend
    pred = blend_weight * pred_xgb + (1-blend_weight) * pred_lstm

    # Generate future timestamps
    from datetime import datetime, timedelta
    now = datetime.now().replace(hour=int(hour.split(":")[0]), minute=0)
    timestamps = [now + timedelta(days=i+1) for i in range(horizon_days)]

    # Scale back to true load units
    pred_vals = scaler_y.inverse_transform(pred.reshape(-1,1)).flatten()
    result = pd.DataFrame({
        "timestamp": timestamps[:horizon_days],
        "demand": pred_vals.repeat(horizon_days)[:horizon_days]    # For demo, flat output
    })
    return result
