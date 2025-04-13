#-------------------------------------------------------- Group 8 Project ---------------------------------------------#

from api_handler import fetch_data_from_api
from ml_model import train_and_predict_LSTM, train_and_predict_linear
import pandas as pd
import os
import time

def generate_rank(coin_map, days, today):

    rank_list = "rank_list.csv"
    rl_df = pd.DataFrame(columns =[ "Coin", "Today ($)", "Tomorrow ($)"])


    for key in coin_map:
        print(key)
        time.sleep(15)
        df = fetch_data_from_api(coin_map[key], days)
        #ML_model_choice = "LSTM"
        current_price = df["price"].iloc[-1]
        model, y_pred, y_test, next_day_price = train_and_predict_LSTM(df, days)
        rl_1 = pd.DataFrame([{"Coin": key, "Today ($)": current_price, "Tomorrow ($)": next_day_price}])
        rl_df = pd.concat([rl_df, rl_1],ignore_index=True)

    #rl_df.to_csv(rank_list, index=False)
    return   rl_df  

'''
rl_df = generate_rank(coin_map, 90, "11-04-2025")
print(rl_df.head(15)) 
'''
