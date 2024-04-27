import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model, initializers,Sequential
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()
df = web.get_data_yahoo('GOOGL', start='2000-01-01')
close = df.Close.values.tolist()
initial_money = 10000
window_size = 30
skip = 1

# class CWDLayer(layers.Layer):
#     def __init__(self,):
#         super(CWDLayer, self).__init__()
#         self.weights = weights
def softmax(X):
    e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
class MasterModel(Model):

    def __init__(self, id_, hidden_size=128):
        super().__init__()

        self.W1 = np.random.randn(window_size, hidden_size) / np.sqrt(window_size)
        self.W2 = np.random.randn(hidden_size, 3) / np.sqrt(hidden_size)
        self.input_layer_flat = layers.Input(shape=(30,),name='in',)
        self.dense1 =layers.Dense(128, activation=tf.nn.relu, name='d1')#,weights=self.W1)
        self.dense2 = layers.Dense(3, activation=tf.nn.softmax,name='d2')
        # assert len(self.dense1.weights ) == 30
        self.fitness = 0
        # print(self.W1,self.W2)
        self.id = id_
    
    def call(self, inputs):
        x = np.dot(inputs,self.W1)
        return softmax(np.dot(np.maximum(x, 0),self.W2))
    def make(self, input_shape=np.zeros((2, 3, 4)))->Model:
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = tf.keras.layers.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='actor')
        print(model.summary())
        return model
# np.zeros(1,3,4)

# model = MasterModel(128)
# model.compile(loss='mse', optimizer='adam')
# print(model.get_build_config(),model.get_metrics_result())
# print(np.argmax(model(np.asarray([close[:30]])), 1)[0])
# print(model.dense2.set_weights(model.W1))
# model = MyModel(0).make((30,))
# print(model.summary(expand_nested=True,show_trainable=True))
# model.dense1.add_weight()
# os.exit()
# Model.

# class NeuralGen(Model):
#     def __init__(self, id_, hidden_size = 128):
#         self.W1 = np.random.randn(window_size, hidden_size) / np.sqrt(window_size)
#         self.W2 = np.random.randn(hidden_size, 3) / np.sqrt(hidden_size)
#         self.fitness = 0
#         print(self.W1,self.W2)
#         self.id = id_

#     def relu(X):
#         return np.maximum(X, 0)
#     def call(self, input):
#         a1 = np.dot(input, self.W1)
#         z1 = relu(a1)
#         a2 = np.dot(z1, self.W2)
#         return tf.nn.softmax(a2)
# def relu(X):
#     return np.maximum(X, 0)
    
# def softmax(X):
#     e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
#     return e_x / np.sum(e_x, axis=-1, keepdims=True)

# def feed_forward(X, nets):
#         #     a = time.time()
#         # self.time_a.append(time.time()-a)
#         # a = time.time()
#         # self.time_b.append(time.time()-a)
#     a1 = np.dot(X, nets.W1)
#     z1 = relu(a1)
#     a2 = np.dot(z1, nets.W2)
#     return softmax(a2)

# class NeuralGen(Model):
#     def __init__(self, hidden_size=128):
#         super(NeuralGen, self).__init__()
        
#         self.W1 = np.random.randn(window_size, hidden_size)/ np.sqrt(window_size)
#         self.W2 = np.random.randn(hidden_size, 3) / np.sqrt(hidden_size)
#         # Input layer is inmplictly declated by the state variable
        
#         self.dense1 = layers.Dense(128, activation='relu',name='first_layer',input_shape=[30,21,29])
#         self.dense2 = layers.Dense(3, activation='softmax',name='output_layer')
#         self.fitness = 0
#         # self.compile(loss='mse', optimizer='adam')
#         # self.dense1.set_weights(self.W1)
        
#     # def build(self, input_shape):
#     #     super(NeuralGen, self).build(input_shape)
        

#     def call(self, inputs):
        
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         return x
    



class NeuroEvolution:
    def __init__(self, population_size, mutation_rate, model_generator,
                state_size, window_size, trend, skip, initial_money):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.model_generator = model_generator
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        self.initial_money = initial_money
        self.time_b = []
        self.time_a = []
        
    def _initialize_population(self):
        self.population = []
        for i in range(self.population_size):
            self.population.append(self.model_generator(i))
    
    def mutate(self, individual, scale=1.0):
        mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=individual.W1.shape)
        individual.W1 += np.random.normal(loc=0, scale=scale, size=individual.W1.shape) * mutation_mask
        mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=individual.W2.shape)
        individual.W2 += np.random.normal(loc=0, scale=scale, size=individual.W2.shape) * mutation_mask
        return individual
    
    def inherit_weights(self, parent, child):
        child.W1 = parent.W1.copy()
        child.W2 = parent.W2.copy()
        return child
    
    def crossover(self, parent1, parent2):
        child1 = self.model_generator((parent1.id+1)*10)
        child1 = self.inherit_weights(parent1, child1)
        child2 = self.model_generator((parent2.id+1)*10)
        child2 = self.inherit_weights(parent2, child2)
        # first W
        n_neurons = child1.W1.shape[1]
        cutoff = np.random.randint(0, n_neurons)
        child1.W1[:, cutoff:] = parent2.W1[:, cutoff:].copy()
        child2.W1[:, cutoff:] = parent1.W1[:, cutoff:].copy()
        # second W
        n_neurons = child1.W2.shape[1]
        cutoff = np.random.randint(0, n_neurons)
        child1.W2[:, cutoff:] = parent2.W2[:, cutoff:].copy()
        child2.W2[:, cutoff:] = parent1.W2[:, cutoff:].copy()
        return child1, child2
    
    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
        res = []
        # print(block)
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        # print(np.array([res]))
        return np.array([res])
    
    def act(self, p, state):
        logits = p(state)
        
        return np.argmax(logits, 1)[0]
    
    def buy(self, individual):
        initial_money = self.initial_money
        starting_money = initial_money
        state = self.get_state(0)
        inventory = []
        states_sell = []
        states_buy = []
        # print(state)
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(individual, state)
            next_state = self.get_state(t + 1)
            
            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f'% (t, self.trend[t], initial_money))
            
            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.trend[t]
                states_sell.append(t)
                try:
                    invest = ((self.trend[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, self.trend[t], invest, initial_money)
                )
            state = next_state
        
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest
    
    def calculate_fitness(self):
        for i in range(self.population_size):
            # start = time.time()
            initial_money = self.initial_money
            starting_money = initial_money
            state = self.get_state(0)

            inventory = []
            a = 0
            for t in range(0, len(self.trend) - 1, self.skip):

                action = self.act(self.population[i], state) # takes the longest?
                next_state = self.get_state(t + 1)

                if action == 1 and starting_money >= self.trend[t]:
                    inventory.append(self.trend[t])
                    starting_money -= self.trend[t]

                elif action == 2 and len(inventory):
                    bought_price = inventory.pop(0)
                    starting_money += self.trend[t]
               
                state = next_state
           
            invest = ((starting_money - initial_money) / initial_money) * 100
            self.population[i].fitness = invest
            # print(time.time()-start)
            # print(i)
      
    def evolve(self, generations=20, checkpoint= 5):
        self._initialize_population()
        n_winners = int(self.population_size * 0.4)
        n_parents = self.population_size - n_winners
        for epoch in range(generations):
            # with cProfile.Profile() as pr:
            self.calculate_fitness()

            # stats.dump_stats("Stats/fast-fitness.prof")
            fitnesses = [i.fitness for i in self.population]
            sort_fitness = np.argsort(fitnesses)[::-1]
            self.population = [self.population[i] for i in sort_fitness]
            fittest_individual = self.population[0]
            if (epoch+1) % checkpoint == 0:
                # print(times)
                print('epoch %d, fittest individual %d with accuracy %f'%(epoch+1, sort_fitness[0], 
                                                                          fittest_individual.fitness))
            next_population = [self.population[i] for i in range(n_winners)]
            total_fitness = np.sum([np.abs(i.fitness) for i in self.population])
            parent_probabilities = [np.abs(i.fitness / total_fitness) for i in self.population]
            parents = np.random.choice(self.population, size=n_parents, p=parent_probabilities, replace=False)

            for i in np.arange(0, len(parents), 2):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                next_population += [self.mutate(child1), self.mutate(child2)]
            self.population = next_population
            
        return fittest_individual

# %%
population_size = 100
generations = 100
mutation_rate = 0.3
neural_evolve = NeuroEvolution(population_size, mutation_rate, MasterModel,
                              window_size, window_size, close, skip, initial_money)
# a = np.random.randn(population_size, 3) / np.sqrt(population_size)
# shape,dtype = a.shape, a.dtype
# print( shape, dtype)

# %%

fittest_nets = neural_evolve.evolve(50)

# %%


# %%
states_buy, states_sell, total_gains, invest = neural_evolve.buy(fittest_nets)


# %%
states_buy

# %%
import datetime as DT

# from pandas import DatetimeIndex
buy_dates = []
# DT.date.st
print(df.index.date[0])
sell_dates = []
start_date = df.head(1).index.date
for i in states_buy:
    date = df.index.date[i]
    buy_dates.append(date)
for j in states_sell:
    date = df.index.date[j]
    sell_dates.append(date)
# print(buy_dates[0:5])
# print(df.iloc[0:5].index)
# df.iloc[df.index.get_loc(buy_dates[0])]


# %%
import plotly.graph_objects as go

import plotly.express as px 
# fig = plt.figure(figsize = (30,15))
# print(df.head(2))
# plt.plot(close, color='r', lw=2.)
# plt.plot(close, '^', markersize=8, color='m', label = 'buying signal', markevery = states_buy)
# plt.plot(close, 'v', markersize=8, color='k', label = 'selling signal', markevery = states_sell)
# plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
# plt.legend()
# plt.show()
# dfpl = df
# xv = list(range(5001))
# fig = go.Figure(data=[go.Scatter(x=xv,y=close)])

# fig.add_scatter(states_buy, mode="markers",
#                 marker=go.scatter.Marker(size=8, color="mediumPurple",angle=0,symbol="arrow"),
#                 name="Buying signal")
# fig.update_layout(xaxis_rangeslider_visible=False)
# fig.show()
# figure = go.Figure(data=[go.Scatter(x=df.index, y=df["Close"],fillcolor="red")])
# figure.add_traces(
#     [
#         figure.add_scatter(
#             x=df[df.index.isin(filter)].index, y=df[df.index.isin(filter)]["Close"]
#         )
#         .update_traces(marker=fmt)
#         .data[0]
#         for filter, fmt in zip(
#             [buy_dates, sell_dates],
#             [
#                 {"color": "black", "symbol": "triangle-up", "size": 10},
#                 {"color": "blue", "symbol": "triangle-down", "size": 10},
#             ],
#         )
#     ]
# )
from network_utils import save_network
save_network(window_size, fittest_nets.W1, fittest_nets.W2, [], 10000)

px.line(df,x=df.index, y=df["Close"]).update_traces(line_color="red").add_traces(
    [
        px.scatter(df[df.index.isin(filter)], y="Close",  width=300, height=150
        )
        .update_traces(marker=fmt)
        .data[0]
        for filter, fmt in zip(
            [buy_dates, sell_dates],
            [
                {"color": "black", "symbol": "triangle-up", "size": 10},
                {"color": "blue", "symbol": "triangle-down", "size": 10},
            ],
        )
    ]
).update_layout(autosize=False,width=2600,height=2400)