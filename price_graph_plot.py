import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from api_handler import fetch_data_from_api
from ml_model import train_and_predict_linear, train_and_predict_LSTM
import pandas as pd

def graph_plot(df, y_pred, y_test, next_day_price):
    # Variables for Graph
    next_day_date = df["date"].max() + pd.Timedelta(days=1)
    test_dates = df["date"].iloc[-len(y_test):]

    # Graph Plot figure
    fig, ax = plt.subplots(figsize=(10, 4))

    #Line graph - Actual Price
    ax.plot(df["date"], df["price"], label="Actual Price")

    #Line graph - Predicted Price Line
    ax.plot(test_dates, y_pred, label="Predicted Price", linestyle='--', marker ="o", markersize=2, markeredgecolor="Black")

    #Point - Prediction Price
    ax.plot(next_day_date, next_day_price, 'ro', label='Predicted Price Next_Day')

    #horizontal Line - Prediction price
    ax.axhline(y=next_day_price, color='red', linestyle=':', label="Predicted P-Line")

    #Graph format
    ax.set_title("bitcoin - Price & Prediction".upper(), loc='center', fontdict={'fontsize': 18, 'fontweight':'bold', 'color':'Black'})
    ax.set_xlabel("Date", fontsize=14, fontweight='bold', color='black')
    ax.set_ylabel("Price (USD $)", fontsize=14, fontweight='bold', color='black')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=90)
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_facecolor('lightgrey')

    # Legend On
    ax.legend()
    ax.grid(True)
    return fig

#Check code
df = fetch_data_from_api("bitcoin", 90)
#model, y_pred, y_test, next_day_price = train_and_predict_linear(df)
model, y_pred, y_test, next_day_price = train_and_predict_LSTM(df, 90)
fig = graph_plot(df, y_pred, y_test, next_day_price)
plt.show()