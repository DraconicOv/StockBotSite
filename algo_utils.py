import numpy as np
import json
import pandas as pd
import pandas_ta as ta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()


def save_strat(symbol, size, TPSLRatio, perc, window, backcandles, trades):
    raise NotImplementedError
def isPivot(candle, window, df):
    """
    function that detects if a candle is a pivot/fractal point
    args: candle index, window before and after candle to test if pivot
    returns: 1 if pivot high, 2 if pivot low, 3 if both and 0 default
    """
    if candle-window < 0 or candle+window >= len(df):
        return 0
    
    pivotHigh = 1
    pivotLow = 2
    for i in range(candle-window, candle+window+1):
        if df.iloc[candle].low > df.iloc[i].low:
            pivotLow=0
        if df.iloc[candle].high < df.iloc[i].high:
            pivotHigh=0
    if (pivotHigh and pivotLow):
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0

def pointpos(x):
    if x['isPivot']==2:
        return x['low']-1e-3
    elif x['isPivot']==1:
        return x['high']+1e-3
    else:
        return np.nan

def detect_structure(candle, backcandles, window, df):
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
df = web.get_data_yahoo('GOOGL',start='2000-01-01')

df.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}, inplace=True)


df=df[df['volume']!=0] # Drop all cols without a volume
df.reset_index(drop=True, inplace=True)

df['EMA'] = ta.ema(df.close, length=50)
df=df[0:]
EMAsignal = [0]*len(df)
backcandles = 10

for row in range(backcandles, len(df)):
    upt = 1
    dnt = 1
    for i in range(row-backcandles, row+1):
        if max(df.open[i], df.close[i])>=df.EMA[i]:
            dnt=0
        if min(df.open[i], df.close[i])<=df.EMA[i]:
            upt=0
    if upt==1 and dnt==1:
        EMAsignal[row]=3
    elif upt==1:
        EMAsignal[row]=216
    elif dnt==1:
        EMAsignal[row]=1

df['EMASignal'] = EMAsignal

df.reset_index(drop=True, inplace=True)
df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
window=10
df['isPivot'] = df.apply(lambda x: isPivot(x.name,window), axis=1)
data = df[:5000].copy()
def SIGNAL():
    return data.pattern_detected
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['RSI'] = ta.rsi(data['Close'])
data.set_index("Date", inplace=True)
data.index = pd.to_datetime(data.index, utc=True)

    
if self.signal!=0 and len(self.trades)==0 and self.data.pattern_detected==2:
    sl1 = self.data.Close[-1]-self.data.Close[-1]*self.perc
    sldiff = abs(sl1-self.data.Close[-1])
    tp1 = self.data.Close[-1]+sldiff*self.TPSLRatio
    tp2 = self.data.Close[-1]+sldiff
    print(tp1,tp2,sl1)
    self.buy(sl=sl1, tp=tp1, size=self.mysize)
    self.buy(sl=sl1, tp=tp2, size=self.mysize)

elif self.signal!=0 and len(self.trades)==0 and self.data.pattern_detected==1:         
    sl1 = self.data.Close[-1]+self.data.Close[-1]*self.perc
    sldiff = abs(sl1-self.data.Close[-1])
    tp1 = self.data.Close[-1]-sldiff*self.TPSLRatio
    tp2 = self.data.Close[-1]-sldiff
    self.sell(sl=sl1, tp=tp1, size=self.mysize)
    self.sell(sl=sl1, tp=tp2, size=self.mysize)