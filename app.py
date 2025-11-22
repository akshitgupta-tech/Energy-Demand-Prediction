import streamlit as st
from vmd_xg_lstm import predict_demand

st.title("India Region-wise Power Demand Forecast (VMD-XG-LSTM)")

HOURS = [f"{h:02d}:00" for h in range(24)]

hour = st.selectbox("Select Hour", options=HOURS)
horizon = st.slider("Prediction Horizon (days)", min_value=1, max_value=31, value=30)

if st.button("Predict"):
    forecast = predict_demand(hour=hour, horizon_days=horizon)
    st.subheader(f"National Demand Prediction at {hour} for Next {horizon} Days")
    st.line_chart(forecast.set_index('timestamp')['demand'])
    st.dataframe(forecast)
    st.download_button("Download CSV", forecast.to_csv(index=False), "forecast.csv")
