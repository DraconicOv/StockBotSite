
import yfinance as yf
import pandas as pd
import os
from pandas import Timestamp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import joblib



def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
        
    return pd.concat(all_predictions)


"""
Precondition: model does not already exist, or should be overwritten (for any reason.) Furthermore, the stock to be analyzed is a conglomorate, like the S&P 500 or the NASDAQ. 
:Parameters:
symbol: the symbol of the stock that is being analyzed
retrain: if the model should be re-trained, using new data (should be true on the first of every month, but that'll be server-side done.)
"""
def make_model(symbol, retrain=False):
    
    if (not retrain) and os.path.exists(f"{symbol}.csv"):
        stock = pd.read_csv(f"{symbol}.csv", index_col=0)
    else:
        stock = yf.Ticker(symbol)
        stock = stock.history(start="1990-01-01") # Large stock-> more time
        stock = stock.drop(columns=["Stock Splits", "Dividends"])
        stock.to_csv(f"{symbol}.csv")

    stock.index = pd.to_datetime(stock.index,utc=True)



    stock["Tomorrow"] = stock["Close"].shift(-1)


    stock["Target"] = (stock["Tomorrow"] > stock["Close"]).astype(int)

    stock = stock.loc[Timestamp.utcfromtimestamp(0):].copy()

    train = stock.iloc[:-100]
    test = stock.iloc[-100:]

    predictors = ["Close", "Volume", "Open", "High", "Low"]

    horizons = [2,5,60,250,1000]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = stock.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        stock[ratio_column] = stock["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        stock[trend_column] = stock.shift(1).rolling(horizon).sum()["Target"]

        new_predictors+= [ratio_column, trend_column]
    stock = stock.dropna(subset=stock.columns[stock.columns != "Tomorrow"])
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    
    predictions = backtest(stock, model, new_predictors)

    # predictions["Predictions"].value_counts()

    # print(precision_score(predictions["Target"], predictions["Predictions"]))

    # predictions["Target"].value_counts() / predictions.shape[0]


    joblib.dump(model, f"{symbol}_random_forest.joblib")


if __name__ == "__main__":
    import sys
    make_model(sys.argv[1])