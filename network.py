import time
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MaxPool1D, Flatten
import keras.layers as layers
from keras.layers import LSTM
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas_ta as ta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()
WINDOW_SIZE = 10
backcandles = 60

df = web.get_data_yahoo('AAPL',start='2000-01-01')



df=df[df['Volume']!=0]
# df.reset_index(drop=True, inplace=True)

df['EMA'] = ta.ema(df.Close, length=50)
# df.reset_index(drop=True, inplace=True)

# EMAsignal = [0]*len(df)
# backcandles = 10

# for row in range(backcandles, len(df)):
#     upt = 1
#     dnt = 1
#     for i in range(row-backcandles, row+1):
#         if max(df.open[i], df.close[i])>=df.EMA[i]:
#             dnt=0
#         if min(df.open[i], df.close[i])<=df.EMA[i]:
#             upt=0
#     if upt==1 and dnt==1:
#         EMAsignal[row]=3
#     elif upt==1:
#         EMAsignal[row]=216
#     elif dnt==1:
#         EMAsignal[row]=1

# df['EMASignal'] = EMAsignal


# def isPivot(candle, window):
#     """
#     function that detects if a candle is a pivot/fractal point
#     args: candle index, window before and after candle to test if pivot
#     returns: 1 if pivot high, 2 if pivot low, 3 if both and 0 default
#     """
#     if candle-window < 0 or candle+window >= len(df):
#         return 0
    
#     pivotHigh = 1
#     pivotLow = 2
#     for i in range(candle-window, candle+window+1):
#         if df.iloc[candle].low > df.iloc[i].low:
#             pivotLow=0
#         if df.iloc[candle].high < df.iloc[i].high:
#             pivotHigh=0
#     if (pivotHigh and pivotLow):
#         return 3
#     elif pivotHigh:
#         return pivotHigh
#     elif pivotLow:
#         return pivotLow
#     else:
#         return 0
# df['isPivot'] = df.apply(lambda x: isPivot(x.name,WINDOW_SIZE), axis=1)

# def pointpos(x):
#     if x['isPivot']==2:
#         return x['low']-1e-3
#     elif x['isPivot']==1:
#         return x['high']+1e-3
#     else:
#         return np.nan
# df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)


def detect_structure(candle, backcandles, window):
    if (candle <= (backcandles+window)) or (candle+window+1 >= len(df)):
        return 0
    
    localdf = df.iloc[candle-backcandles-window:candle-window] #window must be greater than pivot window to avoid look ahead bias
    highs = localdf[localdf['isPivot'] == 1].high.tail(3).values
    lows = localdf[localdf['isPivot'] == 2].low.tail(3).values
    # print(highs)
    # print(lows)
    levelbreak = 0
    zone_width = 0.01
    if len(lows)==3:
        support_condition = True
        mean_low = lows.mean()
        for low in lows:
            if abs(low-mean_low)>zone_width:
                support_condition = True
                # print(abs(low-mean_low))
                break
        # print(support_condition)
        if support_condition and (mean_low - df.loc[candle].close)>zone_width*2:
            levelbreak = 1

    if len(highs)==3:
        resistance_condition = True
        mean_high = highs.mean()
        for high in highs:
            if abs(high-mean_high)>zone_width:
                resistance_condition = True
                break
        if resistance_condition and (df.loc[candle].close-mean_high)>zone_width*2:
            levelbreak = 2
    return levelbreak
print(df.head(5))

#df['pattern_detected'] = df.index.map(lambda x: detect_structure(x, backcandles=40, window=15))
# df['pattern_detected'] = df.apply(lambda row: detect_structure(row.name, backcandles, window=WINDOW_SIZE), axis=1)
df['RSI'] = ta.rsi(df['Close'])
df = df[100:]
# df.set_index("date", inplace=True)
# df.index = pd.to_datetime(df.index, utc=True)
print("DATA READY")
def get_state(df, window_size, t, trend):
    window_size = window_size + 1
    d = t - window_size + 1
    # trend = df.close.values.tolist()
    # block = trend[d : t + 1] if d >= 0 else -d * [trend[0]] + trend[0 : t + 1]
    res = []
    rsis =[]
    ema = []
    c = trend[t:(t+window_size-1),0]
    pivots = []
    # print(block)
    for i in range(window_size - 1):
        # res.append(block[i + 1] - block[i])
        ema.append(df['EMA'].iloc[i])
        rsis.append(df['RSI'].iloc[i])
        # pivots.append(df['isPivot'].iloc[0])
    # print(np.array([res]))
    return np.array([[ema],[rsis],[c]])#,ema,rsis,pivots])

def process_data(df, window_size,skip):
    close = np.asarray(df.Close.values.tolist()).reshape(-1,1)
    # normalize the dataset
    scaler = prep.MinMaxScaler(feature_range=(0, 1))
    close = scaler.fit_transform(close)
    x_data = []
    y_labels =[]
    for t in range(len(close) - window_size -1):
        # a = close[t:(t+window_size),0]
        b = get_state(df,window_size,t,close)
        # print(b.shape)
        x_data.append(b)
        y_labels.append((close[t+window_size, 0] > close[t+window_size-1, 0]) + 2*(close[t+window_size,0] < close[t+window_size-1, 0]))
    return np.asarray(x_data), np.asarray(y_labels)



# def standard_scaler(X_train, X_test):
#     train_samples, train_nx, train_ny = X_train.shape
#     test_samples, test_nx, test_ny = X_test.shape
    
#     X_train = X_train.reshape((train_samples, train_nx * train_ny))
#     X_test = X_test.reshape((test_samples, test_nx * test_ny))
    
#     preprocessor = prep.StandardScaler().fit(X_train)
#     X_train = preprocessor.transform(X_train)
#     X_test = preprocessor.transform(X_test)
    
#     X_train = X_train.reshape((train_samples, train_nx, train_ny))
#     X_test = X_test.reshape((test_samples, test_nx, test_ny))
    
#     return X_train, X_test

# def preprocess_data(stock, seq_len):
#     amount_of_features = len(stock.columns)
#     data = stock.values
    
#     sequence_length = seq_len + 1
#     result = []
#     for index in range(len(data) - sequence_length):
#         result.append(data[index : index + sequence_length])
        
#     result = np.array(result)
#     row = round(0.9 * result.shape[0])
#     train = result[: int(row), :]
    
#     train, result = standard_scaler(train, result)
    
#     X_train = train[:, : -1]
#     y_train = train[:, -1][: ,-1]
#     X_test = result[int(row) :, : -1]
#     y_test = result[int(row) :, -1][ : ,-1]

#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  
#     return [X_train, y_train, X_test, y_test]
# def get_state(self, t):
#     window_size = self.window_size + 1
#     d = t - window_size + 1
#     block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
#     res = []
#     # print(block)
#     for i in range(window_size - 1):
#         res.append(block[i + 1] - block[i])
#     # print(np.array([res]))
#     return np.array([res])

np.random.seed(5)
N_STEPS = 30
x, y = process_data(df, N_STEPS, 1)
# print(x)
x = np.reshape(x,(-1,3,N_STEPS))
print(x[0])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=69) # I am very funny
x_train = np.reshape(x_train,(-1,3,N_STEPS))
# print(x_train[0])
# x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
print("DATASET SPLIT")
# By setting return_sequences to True we are able to stack another LSTM layer
# print(x_train
model = Sequential()
# model.add(layers.Dense(120))
model.add(layers.Conv1D(filters=64, kernel_size=3,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[3, N_STEPS]))
model.add(layers.Bidirectional(layers.LSTM(256,return_sequences=True)))#,input_shape=(N_STEPS, 3)))
model.add(Dropout(0.2))
# cell = layers.LSTMCell(256)
# model.add(layers.RNN(cell=cell))
model.add(layers.Bidirectional(layers.LSTM(256,return_sequences=False)))#, kernel_size=(3)))#kernel_size=(3,N_STEPS))))
# model.add(layers.Dense(120))
model.add(Dropout(0.2))

model.add(Dense(3,activation="linear"))
BATCH_SIZE = 160
EPOCHS = 20
optimizer = keras.optimizers.Adam(learning_rate=0.003)
# optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='huber', optimizer=optimizer,metrics=['accuracy','mse'])
print("MODEL COMPILED")
assert not np.any(np.isnan(x_train))
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1)
model.summary()


print(model.evaluate(x_test,y_test))
model.save("new-model.keras", True)