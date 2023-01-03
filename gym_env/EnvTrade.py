import numpy as np
import pandas
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt


class MaticEnvTrade(gym.Env):
    TRANSACTION_COST_PERCENT = 0.1 / 100

    def __init__(self, df: pandas.DataFrame, initial_balance: int = 10000, time_step=0, previous_state=None):
        super(MaticEnvTrade, self).__init__()

        self.df = df
        self.time_step = time_step
        self.initial_balance = initial_balance

        self.previous_state = previous_state

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,))

        # shape = 6 : Current balance, current price, quantity, macd, rsi, cci

        self.data = self.df.loc[self.time_step, :]
        self.terminal = False

        # initalize state
        self.state = [self.initial_balance] + \
                     [self.data.close] + \
                     [0] + \
                     [self.data.macd] + \
                     [self.data.rsi] + \
                     [self.data.cci]

        # initialize reward and cost
        self.reward = 0
        self.cost = 0

        # memorize all the total balance change
        self.portfolio_value_memory = [self.initial_balance]
        self.rewards_memory = []
        self.trades = 0

        self._seed()

    def _sell(self, amount: int) -> None:
        # sell amount crypto

        if self.state[2] > 0:
            self.state[0] += self.state[1] * min(amount, self.state[2]) * (1 - self.TRANSACTION_COST_PERCENT)
            self.state[2] -= min(amount, self.state[2])
            self.cost += self.state[1] * min(amount, self.state[2]) * self.TRANSACTION_COST_PERCENT
            self.trades += 1
        else:
            pass

    def _buy(self, amount: int) -> None:
        # buy amount crypto

        if self.state[0] >= self.state[1]:
            total_possible = self.state[0] // self.state[1]
            self.state[2] += min(amount, total_possible)
            self.state[0] -= self.state[1] * min(amount, total_possible) * (1 + self.TRANSACTION_COST_PERCENT)
            self.cost += self.state[1] * min(amount, total_possible) * self.TRANSACTION_COST_PERCENT
            self.trades += 1
        else:
            pass

    def step(self, action: int) -> tuple:
        # Execute one time step within the environment
        self.terminal = self.time_step >= len(self.df.index.unique()) - 1

        if self.terminal:
            plt.plot(self.portfolio_value_memory, 'r')
            plt.show()
            plt.plot(self.rewards_memory, 'b')
            plt.show()
            print("=================================")
            print(f"Total Trades: {self.trades}")
            print(f"Total Profit: {self.state[0] + self.state[1] * self.state[2] - self.initial_balance}")
            print("=================================")
            end_total_asset = self.state[0] + self.state[1] * self.state[2]
            return self.state, self.reward, self.terminal, {}

        else:
            if action < 0:
                self._buy(amount=abs(action))
            else:
                self._sell(amount=action)

            self.time_step += 1
            self.data = self.df.loc[self.time_step, :]

            # append portfolio value
            self.portfolio_value_memory.append(self.state[0] + self.state[1] * self.state[2])

            # append reward
            self.rewards_memory.append(self.reward)

            # next state
            self.state = [self.state[0] + self.state[1] * self.state[2] - self.initial_balance] + \
                         [self.data.close] + \
                         [self.state[2]] + \
                         [self.data.macd] + \
                         [self.data.rsi] + \
                         [self.data.cci]

            self.reward = self.state[0] + self.state[1] * self.state[2] - self.initial_balance
            # self.reward -= self.cost

            return self.state, self.reward, self.terminal, {}

    def _reset(self) -> list:
        # Reset the state of the environment to an initial state
        self.state = self.previous_state
        self.time_step = 0
        self.cost = 0
        self.terminal = False
        self.trades = 0
        self.portfolio_value_memory = [self.initial_balance]
        self.rewards_memory = []
        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
