import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests

st.set_page_config(page_title="CryptoPredict AI", layout="centered")
st.title("📈 Crypto Price Prediction using AI")

@st.cache_data
def fetch_data_from_api(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "90",
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["date", "price"]]
    return df

def predict_and_plot(df, coin_name,coin_choice):
    df["day"] = np.arange(len(df))
    X = df[["day"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    next_day = np.array([[df["day"].max() + 1]])
    next_day_price = model.predict(next_day)[0]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["date"], df["price"], label="Actual Price")
    ax.plot(df.iloc[y_test.index]["date"], y_pred, label="Predicted Price", linestyle='--')
    ax.axhline(y=next_day_price, color='red', linestyle=':', label=f"Next Predicted Price: ${next_day_price:.2f}")
    ax.set_title(f"{coin_name} Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.sidebar.header("📊 Dashboard Controls")
    current_price = df["price"].iloc[-1]
    st.metric(label=f"📍 Current {coin_choice.split()[0]} Price (USD)", value=f"${current_price:,.2f}") 
    col1, col2 = st.columns(2)
    col1.metric("📈 Last Price", f"${df['price'].iloc[-1]:.2f}")
    col2.metric("🔮 Predicted Price", f"${next_day_price:.2f}")

    st.success(f"💰 {coin_name} - Predicted Next Day Price: ${next_day_price:.2f}")
    st.markdown("---")
    st.markdown("data programming")
    
# UI
coin_choice = st.selectbox("Select a Cryptocurrency", ["Bitcoin (BTC)", "Ethereum (ETH)"])
if coin_choice == "Bitcoin (BTC)":
    df = fetch_data_from_api("bitcoin")
    predict_and_plot(df, "BTC",coin_choice)
else:
    df = fetch_data_from_api("ethereum")
    predict_and_plot(df, "ETH",coin_choice)