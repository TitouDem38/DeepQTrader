import numpy as np
import pandas
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt


class MaticEnvTrain(gym.Env):
    metadata = {'render.modes': ['human']}
    TRANSACTION_COST_PERCENT = 0.1/100

    def __init__(self, df: pandas.DataFrame, init_balance: int = 10000, timestep: int = 0):
        super(MaticEnvTrain, self).__init__()

        self.timestep = timestep
        self.df = df

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,))

        # shape = 6 : Current balance, current price, quantity, macd, rsi, cci

        self.data = self.df.loc[self.timestep, :]
        self.terminal = False

        # initalize state
        self.state = [self.INITIAL_BALANCE] + \
                     [self.data.close] + \
                     [0] + \
                     [self.data.macd] + \
                     [self.data.rsi] + \
                     [self.data.cci]

        # initialize reward and cost
        self.reward = 0
        self.cost = 0

        # memorize all the total balance change
        self.portfolio_value_memory = [self.INITIAL_BALANCE]
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

        max_amount = self.state[0] // self.state[1]

        self.state[0] -= self.state[1] * min(max_amount, amount) * (1 + self.TRANSACTION_COST_PERCENT)
        self.state[2] += min(max_amount, amount)
        self.cost += self.state[1] * min(max_amount, amount) * self.TRANSACTION_COST_PERCENT
        self.trades += 1

    def step(self, action):
        self.terminal = self.timestep >= len(self.df.index.unique()) - 1

        if not self.terminal:
            initial_portfolio_value = self.state[0] + self.state[1] * self.state[2]

            if action < 0:
                self._sell(amount=abs(action))
            else:
                self._buy(amount=action)

            self.timestep += 1
            self.data = self.df.loc[self.timestep, :]

            # load next state
            self.state = [self.state[0] + self.state[1] * self.state[2]] + \
                         [self.data.close] + \
                         [self.state[2]] + \
                         [self.data.macd] + \
                         [self.data.rsi] + \
                         [self.data.cci]

            # calculate reward
            self.reward = self.state[0] + self.state[1] * self.state[2] - initial_portfolio_value
            self.rewards_memory.append(self.reward)
            self.portfolio_value_memory.append(self.state[0] + self.state[1] * self.state[2])

        else:
            plt.plot(self.portfolio_value_memory, 'r')
            plt.show()

            plt.plot(self.rewards_memory, 'b')
            plt.show()

            print('Total Trades: {}'.format(self.trades))
            print('Total Reward: {}'.format(
                self.state[0] + self.state[1] * self.state[2] - self.INITIAL_BALANCE - self.cost))

            return self.state, self.reward, self.terminal, {}

        return self.state, self.reward, self.terminal, {}

    def _reset(self):
        self.state = [self.INITIAL_BALANCE] + \
                     [self.data.close] + \
                     [0] + \
                     [self.data.macd] + \
                     [self.data.rsi] + \
                     [self.data.cci]

        self.reward = 0
        self.cost = 0
        self.portfolio_value_memory = [self.INITIAL_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        self.terminal = False

        return self.state

    def _render(self, mode='human', close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
