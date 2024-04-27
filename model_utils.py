from collections import deque
import random
import keras
import numpy as np

import pandas_ta as ta
import pandas_datareader.data as web
import yfinance as yf
import sklearn.preprocessing as prep
import pandas_ta as ta
# https://i.redd.it/uch1ua2wb4l41.jpg
yf.pdr_override()
WINDOW_SIZE = 10
backcandles = 60
def a_get_state(df, window_size, t, inventory):
    
    window_size = window_size + 1
    trend = df.Close.values.tolist()
    t = (len(trend)-1)-t
    d = t - window_size + 1
    block = np.asarray(trend[d : t+1] if d >= 0 else -d * [trend[0]] + trend[0 : t+1])

    scaler = prep.MinMaxScaler(feature_range=(0, 1))
    block = scaler.fit_transform(block.reshape(-1,1)).reshape(-1)

    res = np.diff(block)
    inven = scaler.fit_transform((block[:-1]*inventory).reshape(-1,1)).reshape(-1)
    ema = np.diff(df["EMA"].iloc[d:t+1])
    rsis = np.diff(df["RSI"].iloc[d:t+1])#df['RSI'].values[d : t]
    print(res.shape, inven.shape, ema.shape, rsis.shape)
    return np.reshape(np.array([[res],[ema],[rsis],[inven]]),(-1,4,window_size-1))
 # Yes, I leave debug statements in my final code
# def get_state(df, window_size, daysback):
    
#     window_size = window_size + 1
#     trend = df.Close.values.tolist()
#     t = (len(trend)-1)-daysback
#     d = t - window_size + 1
#     block = np.asarray(trend[d : t + 1] if d >= 0 else -d * [trend[0]] + trend[0 : t + 1])
#     scaler = prep.MinMaxScaler(feature_range=(0, 1))
#     block = scaler.fit_transform(block.reshape(-1,1)).reshape(-1)  
#     res = []
#     ema = []
#     rsis = []
#     # print(block)
#     for i in range(window_size - 1):
#         res.append(block[i + 1] - block[i])
#         ema.append(df['EMA'].iloc[d+i])
#         rsis.append(df['RSI'].iloc[d+i])
#     # print(np.array([res]))
#     # Make sure it's the right input shape
#     return np.reshape(np.array([[res],[ema],[rsis]]),(-1,3,window_size-1))
# Fairly self-explanetory, but here goes-- given a point in the dataframe t, return a state containing input information for the model about that point
def get_state(df, window_size, daysback):
    window_size = window_size + 1
    trend = df.Close.values.tolist()
    t = (len(trend)-1)-daysback
    d = t - window_size + 1
    block = np.asarray(trend[d : t + 1] if d >= 0 else -d * [trend[0]] + trend[0 : t + 1])
    scaler = prep.MinMaxScaler(feature_range=(0, 1))
    block = scaler.fit_transform(block.reshape(-1,1)).reshape(-1)  
    res = []
    ema = []
    rsis = []
    # print(block)
    for i in range(window_size - 1):
        res.append(block[i + 1] - block[i])
        ema.append(df['EMA'].iloc[d+i])
        rsis.append(df['RSI'].iloc[d+i])
    # print(np.array([res]))
    # Make sure it's the right input shape
    return np.reshape(np.array([[res],[ema],[rsis]]),(-1,3,window_size-1))

# def get_state(df, window_size, t):
    
#     window_size = window_size + 1
#     d = t - window_size + 1
#     trend = df.Close.values.tolist()
    
#     block = np.asarray(trend[d : t + 1] if d >= 0 else -d * [trend[0]] + trend[0 : t + 1])
#     print(block.shape)
#     scaler = prep.MinMaxScaler(feature_range=(0, 1))
#     block = scaler.fit_transform(block.reshape(-1,1)).reshape(-1)
#     res =[]# np.diff(block)
    
#     ema = []
#     rsis = []#df['RSI'].values[d : t]
#     # print(block)
#     for i in range(window_size - 1):
#         res.append(block[i + 1] - block[i])
#         ema.append(df['EMA'].iloc[d+i])
#         rsis.append(df['RSI'].iloc[d+i])
    
#     # print(np.array([res]))
#     # Make sure it's the right input shape
#     return np.reshape(np.array([[res],[ema],[rsis]]),(-1,3,window_size-1))
class AI_Trader(keras.Model):
  
    def __init__(self, state_size, model_path, num_params = 3, action_space=3,batch_size=32, model_name="AITrader"):
        super().__init__()
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque()
        self.memory_size = 300
        self.batch_size = batch_size
        self.inventory = []
        self.model_name = model_name
        self.num_params = num_params
        self.gamma = 0.99
 
        self.model = keras.saving.load_model(model_path, compile=False)

        self.model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(learning_rate=0.001))


    def trade(self, state):
        actions = self.model(state) # [0.1, 0.7, 0.2], or similar
        return np.argmax(actions, 1)[0] # 1, if we're using the above
    def get_state(self, df, window_size, t):
    
        window_size = window_size + 1
        d = t - window_size + 1
        trend = df.Close.values.tolist()

        block = trend[d : t + 1] if d >= 0 else -d * [trend[0]] + trend[0 : t + 1]
        res = []
        ema = []
        rsis = []
        # print(block)
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
            ema.append(df['EMA'].iloc[d+i])
            rsis.append(df['RSI'].iloc[d+i])
        # print(np.array([res]))
        # Make sure it's the right input shape
        return np.reshape(np.array([[res],[ema],[rsis]]),(-1,3,window_size-1))
    
    def buy(self, df, trend):
        initial_money = 10000
        starting_money = initial_money
        state = get_state(df, self.state_size,0)
        inventory = []
        states_sell = []
        states_buy = []

        for t in range(0, len(trend) - 1, 1):
            action = self.trade(state)
            next_state = get_state(df, self.state_size, t + 1)

            if action == 1 and initial_money >= trend[t]:
                inventory.append(trend[t])
                initial_money -= trend[t]
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f' % (t, trend[t], initial_money))

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += trend[t]
                states_sell.append(t)
                try:
                    invest = ((trend[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, trend[t], invest, initial_money)
                )

            state = next_state

        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        profit = (len(inventory)*trend[-1]-sum(inventory)) + initial_money
        return total_gains, invest, profit

def get_current_price(symbol, daysback=0):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='6mo',interval='1d')
    return todays_data['Close'][-(1+daysback)]


    

def get_action(symbol,daysback=0,holdings=0, balance=10000,history=[]):
    print(symbol + " getting action")
    net = AI_Trader(30, f'{symbol}_trader2.keras')
    net.state_size = net.model.layers[0].input_shape[-1]
    df = web.get_data_yahoo(symbol,start='2010-01-01')
    df=df[df['Volume']!=0] # This confuses the model, and will never happen for stocks I plan to use it for (hopefully)
    df['EMA'] = ta.ema(df.Close, length=50) # Calculate EMA and RSI
    df['RSI'] = ta.rsi(df.Close)
    df = df.iloc[100:]
    if symbol == 'AAPL':
        state = get_state(df, net.state_size, daysback)
    else:
        state = a_get_state(df, net.state_size, daysback, history)
    action = net.trade(state)
    if action == 1:   
        print(f"{symbol} model-network wants to buy")
        if balance < get_current_price(symbol,daysback):
            print(f"{symbol} model-network too broke to buy!")
            return 0
        return 1
    if action == 2:
        print(f"{symbol} model-network wants to sell!")
        if holdings<1:
            print(f"{symbol} model-network can't sell, because it has nothing to sell!")
            return 0
        return 2
    return 0
# symbol = 'GOOGL'
# scaler = prep.MinMaxScaler(feature_range=(0, 1))

# for i in range(10, 70, 10):
    
#     net = AI_Trader(30, f'{symbol}_trader5_{i}.keras')
#     df = web.get_data_yahoo(symbol,start='2020-01-01')
#     df=df[df['Volume']!=0] # This confuses the model, and will never happen for stocks I plan to use it for (hopefully)
#     df['EMA'] = scaler.fit_transform(np.array(ta.ema(df.Close, length=50)).reshape(-1,1)).reshape(-1)
#     df['RSI'] = scaler.fit_transform(np.array(ta.rsi(df.Close )).reshape(-1,1)).reshape(-1)
#     l = len(df.Close.values.tolist())
#     df = df.iloc[-60:]

#     print('data processed')
#     a = net.buy(df,df.Close.values.tolist())
#     print(a)
# DROP TABLE holdings;
# DROP TABLE ACTIONS;
# DROP TABLE modelACTIONS;
# DROP TABLE modelHoldings;
# CREATE TABLE holdings(holdingid INTEGER PRIMARY KEY AUTOINCREMENT, symbol VARCHAR(10) NOT NULL, amount INTEGER NOT NULL);
# CREATE TABLE ACTIONS(actionid INTEGER PRIMARY KEY AUTOINCREMENT, 
#    PRICE DECIMAL(10,2) NOT NULL,
#    symbol VARCHAR(10) NOT NULL,
#    timestampe NOT NULL DEFAULT CURRENT_DATE,
#    action TINYINT,
#    amount INTEGER NOT NULL);
# CREATE TABLE modelACTIONS(actionid INTEGER PRIMARY KEY AUTOINCREMENT, 
#    PRICE DECIMAL(10,2) NOT NULL,
#    symbol VARCHAR(10) NOT NULL,
#    timestampe NOT NULL DEFAULT CURRENT_DATE,
#    action TINYINT,
#    amount INTEGER NOT NULL);
# CREATE TABLE modelHoldings(holdingid INTEGER PRIMARY KEY AUTOINCREMENT, symbol VARCHAR(10) NOT NULL, amount INTEGER NOT NULL);
# INSERT INTO modelACTIONS (PRICE, SYMBOL, ACTION, AMOUNT) VALUES (138.08, 'GOOGL', 0, 0);
# INSERT INTO modelACTIONS (PRICE, SYMBOL, ACTION, AMOUNT) VALUES (189.43, 'AAPL', 0, 0);
# INSERT INTO ACTIONS (PRICE, SYMBOL, ACTION, AMOUNT) VALUES (189.43, 'AAPL', 0, 0);
# INSERT INTO ACTIONS (PRICE, SYMBOL, ACTION, AMOUNT) VALUES (138.08, 'GOOGL', 0, 0);