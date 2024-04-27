from collections import deque
import random
import time
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional,Conv1D, LSTM, Input, Concatenate
import numpy as np
import sklearn.preprocessing as prep
from tqdm import tqdm_notebook, tqdm
from sklearn.model_selection import train_test_split
import backtesting
import pandas_ta as ta
import pandas as pd
import numpy as np
import tensorflow as tf
import pandas_datareader.data as web
import yfinance as yf
# https://i.redd.it/uch1ua2wb4l41.jpg
yf.pdr_override()
WINDOW_SIZE = 10
backcandles = 60
symbol = 'AAPL'

df = web.get_data_yahoo(symbol,start='2015-01-26',interval='1d') # Ari, why are you only using data from 2014-01-01? Because I'm running out of time to finish this, and training time is starting to give me gray hairs!
print(df.tail(5))

df=df[df['Volume']!=0] # This confuses the model, and will never happen for stocks I plan to use it for (hopefully)

# ema = np.asarray(ta.ema(df.Close, length=50)) # Calculate EMA and RSI
# rsi = np.asarray(ta.rsi(df['Close']))
# scaler = prep.MinMaxScaler(feature_range=(0, 1))
# df['EMA'] = scaler.fit_transform(ema.reshape(-1, 1)).reshape(-1)
# df['RSI'] = scaler.fit_transform(rsi.reshape(-1,1)).reshape(-1)
# df = df[100:] # Some values of EMA and RSI can be NaN for the first few values in the dataframe, this fixes that

# Q: Hey Ari, what's EMA, or RSI? 
# A: Not a single clue. I looked up "how do people predict stocks" and these terms showed up. Hopefully, they're helpful to the network.
# I really hope nobody reads this code :(

print("DATA READY") # Yes, I leave debug statements in my final code

# Fairly self-explanetory, but here goes-- given a point in the dataframe t, return a state containing input information for the model about that point
def get_state(df, window_size, t,inventory):
    
    # window_size = window_size + 1
    d = t - window_size
    trend = df.Close.values.tolist()
    block = np.asarray(trend[d : t+1] if d >= 0 else -d * [trend[0]] + trend[0 : t+1])

    scaler = prep.MinMaxScaler(feature_range=(0, 1))
    block = scaler.fit_transform(block.reshape(-1,1)).reshape(-1)

    res = np.diff(block)
    inven = scaler.fit_transform((block[:-1]*inventory).reshape(-1,1)).reshape(-1)
    
    
    ema = np.diff(df["EMA"].iloc[d:t+1])
    rsis = np.diff(df["RSI"].iloc[d:t+1])#df['RSI'].values[d : t]

    return np.reshape(np.array([[res],[ema],[rsis],[inven]]),(-1,4,window_size))


class AI_Trader(keras.Model):
  
    def __init__(self, state_size, num_params = 3, action_space=3,batch_size=32, model_name="AITrader"):
        super().__init__()
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque()
        self.memory_size = 300
        self.batch_size = batch_size
        self.inventory = []
        self.history = [0]*state_size
        self.model_name = model_name
        self.num_params = num_params
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_final = 0.01
        self.model = self.model_builder()
        self.epsilon_decay = 0.05
        self.model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
        
    def model_builder(self):
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=3,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape =[self.num_params, self.state_size]
                      ))
        model.add(Bidirectional(LSTM(256,return_sequences=True)))#,input_shape=(N_STEPS, 3)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(256,return_sequences=False)))#, kernel_size=(3)))#kernel_size=(3,N_STEPS))))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=3, activation='linear'))
        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        
        actions = self.model(state) #  [0.1, 0.7, 0.2], or similar
        action = np.argmax(actions, 1)[0]

        return action
    
    # Memorize information 
    def memorize(self, state, action, reward, new_state, dead):
        self.memory.append((state, action, reward, new_state, dead))
        self.history.append(len(self.inventory))
        if len(self.memory) > self.batch_size:
            # Only remember up to memory-size
            self.memory.popleft()   
        if len(self.history) > self.state_size:
            self.history.pop(0)
        
        
    def replay(self, batch_size):
        
        mini_batch = []
        # Get the correct amount of memories
        mini_batch = random.sample(self.memory,k=batch_size)

        replay_size = len(mini_batch)
        # Start out with empty array
        X = np.empty((replay_size, self.num_params, self.state_size))
        Y = np.empty((replay_size, self.action_space))

        states = np.array([a[0][0] for a in mini_batch])
        new_states = np.array([a[3][0] for a in mini_batch])

        Q = self.model.predict(states, use_multiprocessing=True,verbose=0)
        Q_new = self.model.predict(new_states, use_multiprocessing=True,verbose=0)
        duration_held = deque()
        # Set up x and y values for training
        for i, (state, action, reward, _, done) in enumerate(mini_batch):

            target = Q[i]
            target[action] = reward

            # if not done:
            #     print(np.amax(Q_new[i]))
            #     target[action] += self.gamma * np.amax(Q_new[i])
                # print(target[action])

            X[i] = state
            Y[i] = target

        cost = self.model.fit(X, Y,use_multiprocessing=True,verbose=0)

        # Who's idea was it to call it epsilon, anyway? Nerd.
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

        return cost

    def train(self, iterations, df,skip, base_money, checkpoint=100):
        # we are mostly using the closing values here
        data = df["Close"].values
        data_samples = len(data) - 1
        rep_buy = rep_sell = 0
        
        for epoch in range(1, iterations + 1):
            time_s = time.time()
            print("Episode: {}/{}".format(epoch, iterations))
            total_profit = spending  = contin_pause = rep_sell = total_network_rewards = max_reward = 0
            self.inventory = [data[0]]
            generation_starting_money = base_money-data[0]
            durations = deque()
            state = get_state(df, self.state_size,self.state_size, self.history)
            rep_buy = rep_sell = cont_sell = 0
            prev_profit = prev_nw = previous_reward = rewards = 0
            for t in tqdm(range(self.state_size,data_samples,skip)):
                action = self.trade(state)

                next_state = get_state(df, self.state_size, t+1, self.history)
                reward = 0
                
                if action == 1 and generation_starting_money >= data[t]: #buying
                    contin_pause = rep_sell= 0
                    self.inventory.append(data[t])
                    reward = 1
                #    print("AI Trader bought: ", (data[t]))
                    spending += data[t]
                    # reward += (sum(data[t:t+10])/10 - data[t])*0.1
                    # reward = 0.2+(((sum(data[t:t+self.state_size])/self.state_size)-data[t])*0.4)
                    generation_starting_money -= data[t]
                    durations.append(t)

                elif action == 1:
                    contin_pause = rep_sell = 0
                    reward = 0
                    # reward += min((0,sum(data[t:t+10])/10) - data[t])*0.3

                elif action == 2 and len(self.inventory) > 0: #Selling
                    buy_price = self.inventory.pop(0)
                    contin_pause = rep_sell = 0
                    reward = 1
                    # reward = 0.1 * (data[t] - buy_price)
                    # reward += (data[t]-(sum(data[max(0,int(t-(self.state_size/2))):min(data_samples, int(t+(self.state_size/2)))])/self.state_size))*0.4
                    total_profit += data[t] - buy_price
                    generation_starting_money += data[t]
                elif action == 2:
     
                    contin_pause = rep_sell = 0
                    # Evaluate based on if it's accurate
                    # reward += (data[t]-(sum(data[max(0,int(t-(self.state_size/2))):min(data_samples, int(t+(self.state_size/2)))])/self.state_size))*0.4
                    # reward += min(0, (data[t]-data[t+1])*0.1) 
                    reward = 0
                    # print("AI Trader sold: ", (data[t]), " Profit: " + str((data[t] - buy_price) ))d
                else:
                    contin_pause += 1

                    # Sometimes, the AI will stop doing anything for a long period of time, because not doing anything is enough of a reward
                    # this (hopefully) fixes that
                    reward = -0.5*contin_pause # (previous_reward + (contin_pause*0.1)*data[t+1])*(contin_pause>2)
                    
                if t == data_samples:
                    buy_price = sum(self.inventory)
                    a = data[t]*len(self.inventory) - buy_price
                    generation_starting_money += a
                    total_profit += a
                # reward += 0.3*(((generation_starting_money+(len(self.inventory)*data[t])-sum(self.inventory))-prev_nw) + abs(total_profit-prev_profit) + (0.1*((generation_starting_money+total_profit+(((len(self.inventory)-1)*data[t]-data[t+1])-sum(self.inventory))-base_money)/base_money)))*(rep_sell+contin_pause < 2)*(rep_buy<7)*(cont_sell<7)
                # reward += (((generation_starting_money+total_profit+len(self.inventory)+(((len(self.inventory)-1)*data[t]-data[t+1])-sum(self.inventory)))-base_money)/base_money)*(rep_sell+contin_pause < 2)#*(rep_buy<7)*(cont_sell<7))
                reward += max(((generation_starting_money - base_money) / base_money)*(contin_pause < 0), 0)
                reward += max(0,((((data[t]*len(self.inventory))-sum(self.inventory))+generation_starting_money)/base_money)*0.1*(contin_pause < 0))
                
                previous_reward = reward 
                # prev_profit = total_profit
                # prev_nw = generation_starting_money+(len(self.inventory)*data[t])-sum(self.inventory)
                # if abs(reward) >= max_reward or abs(reward) > 20:
                #     max_reward = abs(reward)
                #     print(action)
                #     print(reward)
                #     print(data[t],t)
                #     print(contin_pause, rep_sell)
                # total_network_rewards += rewards
                # if rep_sell+contin_pause > 2 and reward > 0 or rep_sell+contin_pause > 30:
                #     print(reward, rep_sell, contin_pause, action)
                #     reward = min(reward, 0)
                # Remember what happened here
                self.memorize(state, action, reward, next_state, generation_starting_money<base_money)

                state = next_state

                if reward < -1 and t % 5 == 0:
                    print(action, reward, rep_sell, rep_buy, cont_sell, contin_pause)
            
                if t == data_samples - 1:
                    print("########################")
                    print("TOTAL PROFIT: {}".format(total_profit))
                    print("########################")

                if len(self.memory) >= self.batch_size: # Chosen at random, tbh
                    cost = self.replay(self.batch_size)
               
                if (t+1) % checkpoint == 0:
                    print('generation: %d, total liquid profits: %f.3, total rewards: %f.3, cost: %f, total money: %f, amount held: %d, net worth: %f'%(t + 1, total_profit, total_network_rewards, spending,
                                                                                 generation_starting_money, len(self.inventory),generation_starting_money+(len(self.inventory)*data[t])))
            if epoch % 10 == 0:
                self.model.save(symbol+"_trader6_{}.keras".format(epoch))
            print(time.time()-time_s)
window_size = 16
episodes = 1000
data = df["Close"].values.tolist()
batch_size = 32
scaler = prep.MinMaxScaler(feature_range=(0, 1))
df['EMA'] = scaler.fit_transform(np.array(ta.ema(df.Close, length=10)).reshape(-1,1)).reshape(-1)
df['RSI'] = scaler.fit_transform(np.array(ta.rsi(df.Close,length=10 )).reshape(-1,1)).reshape(-1)

# df['EMA'] = ta.ema(df.Close, length=50)
# df['RSI'] = ta.rsi(df['Close'])   
df = df[100:]
trend = df.Close.values.tolist()
data_samples = len(data) - 1
trader = AI_Trader(window_size,batch_size=batch_size,num_params=4)
trader.model.summary()
trader.train(iterations=1000,df=df,skip=1,base_money=10000)

