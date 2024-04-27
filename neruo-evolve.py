# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cProfile
import pstats
sns.set()

import pandas_datareader.data as web
import yfinance as yf
# %%
yf.pdr_override()
df = web.get_data_yahoo('GOOGL',start='2000-01-01')
df.head()

# %%
close = df.Close.values.tolist()
# print(len(close))
# print(close[0
initial_money = 10000
window_size = 30
skip = 1

# %%
import tensorflow as tf
import numpy as np
import keras
import time
import timeit
import sys
class NetworkBase(keras.Model):
    def __init__(self,id_:int, hidden_size=128)->keras.Model:
        super(NetworkBase, self).__init__(id)
        
        # self.W1 = tf.Variable(np.random.randn(window_size, hidden_size)/ np.sqrt(window_size))
        # self.W2 = tf.Variable(np.random.randn(hidden_size, 3) / np.sqrt(hidden_size))
        
        
        self.W1 = tf.Variable(tf.math.divide(tf.random.normal((window_size, hidden_size),dtype=tf.float32),tf.sqrt(tf.constant([window_size],dtype=tf.float32))))
        self.W2 = tf.Variable(tf.math.divide(tf.random.normal((hidden_size, 3),dtype=tf.float32), tf.sqrt(tf.constant([hidden_size],dtype=tf.float32))))
        self.fitness = 0
        self.id = id_
        
    def call(self, inputs) -> int:

        a1 = tf.matmul(inputs,self.W1)#np.dot(inputs.numpy(), self.W1.numpy())
        
        z1 = tf.nn.relu(a1)
        # a2 = np.dot(z1.numpy(), self.W2.numpy())
        a2 = tf.matmul(z1,self.W2)
        
        return tf.nn.softmax(a2)
        

def feed_forward(network, state):
    # print(network, state)
    return network(state)
import keras

class NeuroEvolution(keras.Model):
    def __init__(self, population_size:int, mutation_rate:float, model_generator:NetworkBase, state_size, window_size:int|float, trend:list, skip:int|float, initial_money:float):
        super(NeuroEvolution, self).__init__()
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.model_generator = model_generator
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = tf.convert_to_tensor(trend)
        self.skip = skip
        self.initial_money = initial_money

    def _initialize_population(self):
        self.population: list[NetworkBase] = [self.model_generator(id_=_) for _ in range(self.population_size)]

    def mutate(self, individual: NetworkBase, scale=1.0):
        mutation_mask = tf.cast(tf.random.uniform(individual.W1.shape) < self.mutation_rate, dtype=tf.float32)
        individual.W1.assign_add(tf.random.normal(shape=individual.W1.shape, stddev=scale) * mutation_mask)

        mutation_mask = tf.cast(tf.random.uniform(individual.W2.shape) < self.mutation_rate, dtype=tf.float32)
        individual.W2.assign_add(tf.random.normal(shape=individual.W2.shape, stddev=scale) * mutation_mask)

        return individual

    def inherit_weights(self, parent: NetworkBase, child: NetworkBase)->NetworkBase:
        child.W1.assign(parent.W1.numpy())
        child.W2.assign(parent.W2.numpy())
        return child
    
    def crossover(self, parent1, parent2)->tuple[NetworkBase, NetworkBase] :
        child1 = self.model_generator(id_=(parent1.id + 1) * 10)
        child2 = self.model_generator(id_=(parent2.id + 1) * 10)
        

        n_neurons = child1.W1.shape[1]
        cutoff = np.random.randint(0, n_neurons)
        child1.W1[:, cutoff:].assign(parent2.W1[:, cutoff:].numpy())
        child2.W1[:, cutoff:].assign(parent1.W1[:, cutoff:].numpy())

        n_neurons = child1.W2.shape[1]
        cutoff = np.random.randint(0, n_neurons)
        child1.W2[:, cutoff:].assign(parent2.W2[:, cutoff:].numpy())
        child2.W2[:, cutoff:].assign(parent1.W2[:, cutoff:].numpy())

        return child1, child2
    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
        res = []
        for i in range(window_size-1):
            res.append(block[i + 1] - block[i])
        return tf.constant([res])
    
    # def get_state(self, t):
    #     window_size = self.window_size + 1
    #     d = max(0, t - window_size + 1) 
    #     block = self.trend[d: t + 1] + [self.trend[-1]] * max(0, window_size - len(self.trend[d: t + 1]))
    #     res = [block[i + 1] - block[i] for i in range(window_size - 1)]
    #     return np.array([res])
    @tf.function
    def act(self, network, state):
        logits = feed_forward(network, state)
        return tf.argmax(logits, 1)
        # return np.argmax(logits, 1).numpy()[0]

    def buy(self, individual):
        initial_money = self.initial_money
        starting_money = initial_money
        state = self.get_state(0)
        inventory = []
        states_sell = []
        states_buy = []

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(individual, state)
            next_state = self.get_state(t + 1)

            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f' % (t, self.trend[t], initial_money))

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
    
    @tf.function
    def calculate_fitness(self):
        # print(self.population_size)
        tf_range = tf.range(self.population_size)
        print(tf_range)
        for i in tf.range(self.population_size):
            initial_money = tf.constant(self.initial_money)
            starting_money = initial_money
            state = self.get_state(0)
            print(tf.get_static_value(i))
            inventory = tf.TensorArray(dtype=tf.float32,size=self.trend)
            print(i)
            print("Fit_calc_established")
            for t in tf.range(0, len(self.trend) - 1, self.skip):
                action = self.act(self.population[i], state)
                next_state = self.get_state(t + 1)
                # print(time.time()-time_start)

                if action == 1 and starting_money >= self.trend[t]:
                    inventory.write(self.trend[t])
                    starting_money -= self.trend[t]

                elif action == 2 and len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    starting_money += self.trend[t]

                state = next_state
            print("actions set")
            invest = ((starting_money - initial_money) / initial_money) * 100
            self.population[i].fitness = invest
            print(i)
        print("done")
        
    # @tf.function
    def evolve(self, generations=20, checkpoint=5):
        start_time = time.time()
        self._initialize_population()
        n_winners = int(self.population_size * 0.4)
        n_parents = self.population_size - n_winners
        for epoch in range(generations):
            epoch_time = time.time()
            # print(epoch)
            with cProfile.Profile() as pr:
                self.calculate_fitness()
            pr.dump_stats("Stats/test.prof")
            print("fitness")
            fitnesses = [i.fitness for i in self.population]
            sort_fitness = np.argsort(fitnesses)[::-1]
            self.population = [self.population[i] for i in sort_fitness]
            fittest_individual = self.population[0]
            print('epoch %d, fittest individual %d with accuracy %f' % (epoch + 1, sort_fitness[0],
                                                                            fittest_individual.fitness))
            if (epoch + 1) % checkpoint == 0:
                print('epoch %d, fittest individual %d with accuracy %f' % (epoch + 1, sort_fitness[0],
                                                                            fittest_individual.fitness))
            next_population = [self.population[i] for i in range(n_winners)]
            total_fitness = np.sum([np.abs(i.fitness) for i in self.population])
            parent_probabilities = [np.abs(i.fitness / total_fitness) for i in self.population]
            parents = np.random.choice(self.population, size=n_parents, p=parent_probabilities, replace=False)
            for i in np.arange(0, len(parents), 2):
                child1, child2 = self.crossover(parents[i], parents[i + 1])
                next_population += [self.mutate(child1), self.mutate(child2)]
            self.population = next_population
            print(f'Epoch {i} finished in {time.time()-epoch_time}')
        print(f'Total time for {generations} epochs is {time.time()-start_time}')
        return fittest_individual

# Assuming the rest of your code remains the same

# Example usage:
population_size = 5
generations = 100
mutation_rate = 0.3
window_size = 30  # Replace with the appropriate value
# close = np.random.randn(window_size)  # Replace with your actual data
skip = 1  # Replace with the appropriate value
initial_money = 10000  # Replace with the appropriate value

neural_evolve = NeuroEvolution(population_size, mutation_rate, NetworkBase,
                              window_size, window_size, close, skip, float(initial_money))

fittest_nets = neural_evolve.evolve(2)

print(type(fittest_nets))


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


# %%



