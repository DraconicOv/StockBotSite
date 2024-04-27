import numpy as np
import pandas_datareader.data as web
import yfinance as yf
from json import JSONEncoder
import json
yf.pdr_override()
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Net():
    def __init__(self, w1, w2):
        self.W1 = w1 # weights
        self.W2 = w2
    
    def call(self, input):
        dense_layers = np.dot(np.maximum(np.dot(input,self.W1), 0),self.W2)
        soft = np.exp(dense_layers-np.max(dense_layers, axis=-1, keepdims=True))
        return soft/np.sum(soft,axis=-1, keepdims=True)
    
def get_state(df, window_size, daysback):
    window_size = window_size + 1
    trend = df.Close.values.tolist()
    t = (len(trend)-1)-daysback
    d = t - window_size + 1
    block = trend[d : t + 1] if d >= 0 else -d * [trend[0]] + trend[0 : t + 1]
    res = []
    for i in range(window_size - 1):
        res.append(block[i + 1] - block[i])
    return np.array([res])

def get_current_price(symbol, daysback=0):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='6mo',interval='1d')
    return todays_data['Close'][-(1+daysback)]

def act(individual:Net, state):
    action = np.argmax(individual.call(state),1)[0]

    return action

def save_network(symbol, window_size, W1, W2, holdings, balance):
    network_info = {"window_size":window_size,"W1":W1,"W2":W2,"holdings":holdings,"balance":balance}
    with open(f'{symbol}-evolved-network.json', 'w') as fp:
        json.dump(network_info, fp,  cls=NumpyEncoder)
    

def get_action(symbol,daysback=0,holdings=0,balance=10000):
    with open(f'{symbol}-evolved-network.json', 'r') as fp:
        network = json.load(fp)
    net = Net(np.asarray(network['W1']), np.asarray(network['W2']))
    df = web.get_data_yahoo(symbol,start='2000-01-01')
    state = get_state(df, network['window_size'],daysback)
    action = act(net, state)
    if action == 1:   
        print(f"{symbol} network wants to buy")
        if balance < get_current_price(symbol,daysback):
            print(f"{symbol} network too broke to buy!")
            return 0
        return 1
    if action == 2:
        print(f"{symbol} network wants to sell!")
        if holdings<1:
            print(f"{symbol} network can't sell, because it has nothing to sell!")
            return 0
        return 2
    return 0

