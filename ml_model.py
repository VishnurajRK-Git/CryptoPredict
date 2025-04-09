from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def train_and_predict(df):
    X = df[["day"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    next_day = np.array([[df["day"].max() + 1]])
    next_day_price = model.predict(next_day)[0]
    
    return model, y_pred, y_test, next_day_price
