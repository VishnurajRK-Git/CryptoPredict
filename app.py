import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from price_graph_plot import graph_plot
from api_handler import fetch_data_from_api
from ml_model import train_and_predict_LSTM, train_and_predict_linear
from Rank_List_data import generate_rank
import pandas as pd
import os

st.set_page_config(page_title="CryptoPredict AI", layout="centered")
st.title("📈 CryptoPredict AI")
#st.sidebar.subheader()
st.sidebar.markdown(
    "<h3 style='font-size:24px; font-weight:600; color:#1E90FF;'>Group 08</h3>",
    unsafe_allow_html=True
)
coin_map = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Tether": "tether",
    "Solana": "solana",
    "BinanceCoin": "binancecoin",
    "RippleXRP": "ripple",
    "Dogecoin": "dogecoin",
    "Tron": "tron",
    "Cardano": "cardano",
    "Sui": "sui"
}

coin_choice = st.sidebar.selectbox("Select Cryptocurrency", list(coin_map.keys()))
days = st.sidebar.selectbox("Select number of days", [7, 30, 90], index = 2)
ML_model_choice = st.sidebar.selectbox("Select Prediction Model", ("Linear Regression", "LSTM"), index=1)
coin_id = coin_map[coin_choice]

st.subheader(coin_choice.upper())

df = fetch_data_from_api(coin_id, days)

if ML_model_choice == "Linear Regression":
    model, y_pred, y_test, next_day_price = train_and_predict_linear(df)
elif ML_model_choice == "LSTM":
    model, y_pred, y_test, next_day_price = train_and_predict_LSTM(df, days)

# Log write of Prediction data
with open("predictions.csv", "a") as f:
    f.write(f"{coin_choice},{next_day_price:.2f},{ML_model_choice},{pd.Timestamp.now()}\n")
current_price = df["price"].iloc[-1]



# Values display
current_price = df["price"].iloc[-1]
st.metric("📍 Current Price", f"${current_price:,.2f}")
st.metric("🔮 Predicted Next Day", f"${next_day_price:,.2f}")

#plot graph
fig = graph_plot(df, y_pred, y_test, next_day_price)
st.pyplot(fig)

# Log of previous prediction
if os.path.exists("predictions.csv"):
    st.subheader("Previous Predictions")
    df_logs = pd.read_csv("predictions.csv", header=None, names=["Coin", "Predicted Price", "Prediction_model", "Timestamp"])
    st.dataframe(df_logs.tail(3))
else:
    st.info("No previous predictions found yet.")


# Populate the prediction of all coin data and show.

today = df["date"].max().date()
rank_list = "rank_list_" + str(today) + ".csv"

# Arrow for conditional formating
def arrow_marker(actual, prediction):
    if prediction > actual:
        return '🟢 ⬆️'
    elif prediction < actual:
        return '🔴 ⬇️'
    else:
        return '➖'

st.sidebar.subheader("Ranking Crypto Currency")
status_run = st.sidebar.empty()
if os.path.exists(rank_list):
    r_df = pd.read_csv(rank_list, header = 0)
else:
    status_run.info("Analyzing & Generating!!!!")
    rl_df = generate_rank(coin_map, days, today)
    rl_df.to_csv(rank_list, index=False)
    r_df = pd.read_csv(rank_list, header = 0)

r_df = r_df.dropna(axis=1, how='all')
r_df = r_df.reset_index(drop=True)
r_df = r_df.sort_values(by="Tomorrow ($)", ascending=False)

r_df["Trend"] = [arrow_marker(a, p) for a, p in zip(r_df["Today ($)"], r_df["Tomorrow ($)"])]
#r_df["Trend"] = r_df.apply(lambda row: arrow_marker(row['Actual price'], row['Prediction price']), axis=1)

styled_df = r_df[["Coin","Today ($)", "Tomorrow ($)", "Trend"]].style.hide(axis="index").set_properties(**{
    'font-size': '14px',
    'text-align': 'Center',
    'padding': '2px'
}).format({"Today ($)": "${:,.2f}", "Tomorrow ($)": "${:,.2f}"})
st.sidebar.markdown(styled_df.to_html(escape=False), unsafe_allow_html=True)
status_run.empty()