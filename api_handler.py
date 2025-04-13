#-------------------------------------------------------- Group 8 Project ---------------------------------------------#
# API data extraction
import pandas as pd
import numpy as np
import requests

def fetch_data_from_api(coin_id,days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": {days}, "interval": "daily"}
    df = pd.DataFrame()
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise error if not 200

        data = response.json()
        
        # Check if 'prices' key exists
        if "prices" not in data:
            print(f"'prices' key not found for {coin_id}. Response: {data}")
            return df  # Return empty DataFrame
    
    
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["date", "price"]]
        df["day"] = np.arange(len(df))
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {coin_id}: {e}")
        return df
'''
#Code check
df = fetch_data_from_api('bitcoin',90)
print(df.tail(10))


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

for key in coin_map:
    print(key)
    df = fetch_data_from_api(coin_map[key],90)
    print(df.tail(5))
'''