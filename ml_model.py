#-------------------------------------------------------- Group 8 Project ---------------------------------------------#
#Machine Learning Model
from api_handler import fetch_data_from_api
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# Linear Regression Model
def train_and_predict_linear(df):
    X = df[["day"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  #Split Data for train and test
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    next_day = np.array([[df["day"].max() + 1]])
    next_day_price = model.predict(next_day)[0]

    return model, y_pred, y_test, next_day_price
'''
#Check Code
df = fetch_data_from_api("bitcoin", 30)
model, y_pred, y_test, next_day_price = train_and_predict_linear(df)
print(y_test)
print(y_pred)
print("next_day_price:", next_day_price)
'''

# Long-Short Term Model (LSTM)
def train_and_predict_LSTM(df, days_tot):
    lstm_data_price = df[['price']].values
    scaler = MinMaxScaler() 
    scaled_prices = scaler.fit_transform(lstm_data_price)

    if days_tot == 7:
        sequence_length = 3
    else:
        sequence_length = 6

    X, y = [], []

    for i in range(len(scaled_prices) - sequence_length):
        X.append(scaled_prices[i:i + sequence_length])
        y.append(scaled_prices[i + sequence_length])

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    #LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(sequence_length, 1)))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=2, verbose=0)

    #Predict and inverse transform
    predicted_price_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(predicted_price_scaled)
    y_test_orig = scaler.inverse_transform(y_test)

    last_sequence = scaled_prices[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, 1))
    next_day_scaled = model.predict(last_sequence)
    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]

    return model, y_pred.flatten(), y_test_orig.flatten(), next_day_price

'''
#Check Code
df = fetch_data_from_api("bitcoin", 30)
model, y_pred, y_test, next_day_price = train_and_predict_LSTM(df,30)
print(y_test)
print(y_pred)
print("next_day_price:", next_day_price)
'''