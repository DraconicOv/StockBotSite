import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
import pandas_ta as pa
import os
import joblib
import yfinance as yf

def mytarget(barsupfront, df1, pipdiff=250*1e-4, SLTPRatio=1):
    length = len(df1)
    high = list(df1['High'])
    low = list(df1['Low'])
    close = list(df1['Close'])
    open = list(df1['Open'])
    trendcat = [None] * length
    for line in range (0,length-barsupfront-2):
        valueOpenLow = 0
        valueOpenHigh = 0
        for i in range(1,barsupfront+2):
            value1 = open[line+1]-low[line+i]
            value2 = open[line+1]-high[line+i]
            valueOpenLow = max(value1, valueOpenLow)
            valueOpenHigh = min(value2, valueOpenHigh)
            if ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
                trendcat[line] = 1 #-1 downtrend
                break
            elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ):
                trendcat[line] = 2 # uptrend
                break
            else:
                trendcat[line] = 0 # no clear trend
            
    return trendcat
def support(df1, l, n1, n2): #n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if(df1.Low[i]>df1.Low[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.Low[i]<df1.Low[i-1]):
            return 0
    return 1

def resistance(df1, l, n1, n2): #n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if(df1.High[i]<df1.High[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.High[i]>df1.High[i-1]):
            return 0
    return 1
def isEngulfing(l, bodydiff, close, open):
    row=l
    bodydiff[row] = abs(open[row]-close[row])
    if bodydiff[row]<0.000001:
        bodydiff[row]=0.000001      

    bodydiffmin = 0.002
    if (bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
        open[row-1]<close[row-1] and
        open[row]>close[row] and 
        (open[row]-close[row-1])>=-0e-5 and close[row]<open[row-1]): #+0e-5 -5e-5
        return 1

    elif(bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
        open[row-1]>close[row-1] and
        open[row]<close[row] and 
        (open[row]-close[row-1])<=+0e-5 and close[row]>open[row-1]):#-0e-5 +5e-5
        return 2
    else:
        return 0
       
def isStar(l, lowdiff, highdiff, high, low, close, bodydiff, ratio1, ratio2, open):
    bodydiffmin = 0.0020
    row=l
    highdiff[row] = high[row]-max(open[row],close[row])
    lowdiff[row] = min(open[row],close[row])-low[row]
    bodydiff[row] = abs(open[row]-close[row])
    if bodydiff[row]<0.000001:
        bodydiff[row]=0.000001
    ratio1[row] = highdiff[row]/bodydiff[row]
    ratio2[row] = lowdiff[row]/bodydiff[row]

    if (ratio1[row]>1 and lowdiff[row]<0.2*highdiff[row] and bodydiff[row]>bodydiffmin):# and open[row]>close[row]):
        return 1
    elif (ratio2[row]>1 and highdiff[row]<0.2*lowdiff[row] and bodydiff[row]>bodydiffmin):# and open[row]<close[row]):
        return 2
    else:
        return 0
    
def closeResistance(l,levels,lim, df):
    if len(levels)==0:
        return 0
    c1 = abs(df.High[l]-min(levels, key=lambda x:abs(x-df.High[l])))<=lim
    c2 = abs(max(df.Open[l],df.Close[l])-min(levels, key=lambda x:abs(x-df.High[l])))<=lim
    c3 = min(df.Open[l],df.Close[l])<min(levels, key=lambda x:abs(x-df.High[l]))
    c4 = df.Low[l]<min(levels, key=lambda x:abs(x-df.High[l]))
    if( (c1 or c2) and c3 and c4 ):
        return 1
    else:
        return 0
    
def closeSupport(l,levels,lim, df):
    if len(levels)==0:
        return 0
    c1 = abs(df.Low[l]-min(levels, key=lambda x:abs(x-df.Low[l])))<=lim
    c2 = abs(min(df.Open[l],df.Close[l])-min(levels, key=lambda x:abs(x-df.Low[l])))<=lim
    c3 = max(df.Open[l],df.Close[l])>min(levels, key=lambda x:abs(x-df.Low[l]))
    c4 = df.High[l]>min(levels, key=lambda x:abs(x-df.Low[l]))
    if( (c1 or c2) and c3 and c4 ):
        return 1
    else:
        return 0
    
def prep_df(df):
    length = len(df)
    high = list(df['High'])
    low = list(df['Low'])
    close = list(df['Close'])
    open = list(df['Open'])
    bodydiff = [0] * length
    
    highdiff = [0] * length
    lowdiff = [0] * length
    ratio1 = [0] * length
    ratio2 = [0] * length
        
    n1=2
    n2=2
    backCandles=30
    signal = [0] * length
    
    for row in range(backCandles, len(df)-n2):
        ss = []
        rr = []
        for subrow in range(row-backCandles+n1, row+1):
            if support(df, subrow, n1, n2):
                ss.append(df.Low[subrow])
            if resistance(df, subrow, n1, n2):
                rr.append(df.High[subrow])
        #!!!! parameters
        if ((isEngulfing(row, bodydiff, close, open)==1 or isStar(row, lowdiff, highdiff, high, low, close, bodydiff, ratio1, ratio2, open)==1) and closeResistance(row, rr, 150e-5,df) ):
            signal[row] = 1
        elif((isEngulfing(row, bodydiff, close, open)==2 or isStar(row, lowdiff, highdiff, high, low, close, bodydiff, ratio1, ratio2, open)==2) and closeSupport(row, ss, 150e-5, df)):
            signal[row] = 2
        else:
            signal[row] = 0
    
    return signal
"""
Precondition: model does not already exist, or should be overwritten (for any reason.) Furthermore, the stock to be analyzed is not a conglomorate (like the  S&P 500) or cryptocurrency (like bitcoin). 
:Parameters:
symbol: the symbol of the stock that is being analyzed
retrain: if the model should be re-trained, using new data (should be true on the first of every month, but that'll be server-side done.)
"""
def make_model(symbol, retrain, start=1985):
    if (not retrain) and os.path.exists(f"{symbol}.csv"):
        df = pd.read_csv(f"{symbol}.csv")
    else:
        stock = yf.Ticker(symbol)
        stock = stock.history(start=f"{str(start)}-01-01") # for smaller stocks, less data works a little better-- and less training time
        
        stock = stock[["Open","High","Low","Close","Volume"]]
        stock.to_csv(f"{symbol}.csv")
    df = pd.read_csv(f"{symbol}.csv")

    #Check if NA values are in data (remove if there are)
    df=df[df['Volume']!=0]
    df.reset_index(drop=True, inplace=True)
    
    df['Signal'] = prep_df(df)
    print(len(df[df['Signal']==0]))

    
    df.columns = ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume','signal']
    def SIGNAL(): #TODO: check if this is even used, wtf past Ari?
        return df.signal # this really should be outside of this function, but eh.
    
    #Target flexible way
    pipdiff = 250*1e-4 #for TP
    SLTPRatio = 1 #pipdiff/Ratio gives SL
    
    
    
    #!!! pitfall one category high frequency
    df['Target'] = mytarget(30, df)  
    df['Target'].hist()  
    
    df["RSI"] = pa.rsi(df.Close, length=16)
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)    
    
    attributes = ['RSI', 'signal', 'Target']
    df_model= df[attributes].copy()
    df_model['signal'] = pd.Categorical(df_model['signal'])
    dfDummies = pd.get_dummies(df_model['signal'], prefix = 'signalcategory')
    df_model= df_model.drop(['signal'], axis=1)
    df_model = pd.concat([df_model, dfDummies], axis=1)

    # It's somewhat random if XGB or MLP will be better, so we use both and see which works better for a given stock.
    
    attributes = ['RSI', 'signalcategory_0', 'signalcategory_1', 'signalcategory_2']
    X = df_model[attributes]
    y = df_model['Target']
    
    train_pct_index = int(0.7 * len(X))
    X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]

    model = XGBClassifier()
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test) 
    pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    model_0_acc_test = accuracy_score(y_test, pred_test)
    for i in pred_test:
        if i !=0:
            print(i)
    # for debugging, print if needed
    print(model_0_acc_test)
    print('****Model 0 Train Results****')
    print("Accuracy: {:.4%}".format(acc_train))
    print('****Model 0 Test Results****')
    print("Accuracy: {:.4%}".format(model_0_acc_test))
    train_pct_index = int(0.6 * len(X))
    X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]
    if model_0_acc_test < 0.6 and start < 2015:
        return make_model(symbol, True, start+7)
    
    model.save_model(f'{symbol}_xgboost_model.json')
    print(start)
    return 0

if __name__== "__main__":
    import sys
    make_model(sys.argv[1], True, 1985) #for testing