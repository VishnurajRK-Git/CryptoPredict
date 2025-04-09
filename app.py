import streamlit as st
import matplotlib.pyplot as plt
from api_handler import fetch_data_from_api
from ml_model import train_and_predict
import pandas as pd
import os

st.set_page_config(page_title="CryptoPredict AI", layout="centered")
st.title("📈 Crypto Price Prediction")


coin_map = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum"
}
coin_choice = st.sidebar.selectbox("Select Cryptocurrency", list(coin_map.keys()))
days = st.sidebar.selectbox("Select number of days", [7, 30, 90])
coin_id = coin_map[coin_choice]

df = fetch_data_from_api(coin_id, days)

model, y_pred, y_test, next_day_price = train_and_predict(df)
with open("predictions.csv", "a") as f:
    f.write(f"{coin_choice},{next_day_price:.2f},{pd.Timestamp.now()}\n")
current_price = df["price"].iloc[-1]

st.metric("📍 Current Price", f"${current_price:,.2f}")
st.metric("🔮 Predicted Next Day", f"${next_day_price:,.2f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["date"], df["price"], label="Actual Price")
ax.plot(df.iloc[y_test.index]["date"], y_pred, label="Predicted", linestyle='--')
ax.axhline(y=next_day_price, color='red', linestyle=':', label="Predicted Next")
ax.legend()
ax.grid(True)
st.pyplot(fig)
if os.path.exists("predictions.csv"):
    st.subheader("Previous Predictions")
    df_logs = pd.read_csv("predictions.csv", header=None, names=["Coin", "Predicted Price", "Timestamp"])
    st.dataframe(df_logs.tail(3))
else:
    st.info("No previous predictions found yet.")
    

