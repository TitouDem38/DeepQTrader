from Agent import Agent
from gym_env.EnvTrain import MaticEnvTrain
from gym_env.EnvTrade import MaticEnvTrade

import pandas as pd
import numpy as np
import gym
import time


class DeepQTrader:

    def __init__(self):
        self.agents = []
        self.training_env = {}
        self.trading_env = {}

        self.retrain_window = 0
        self.validation_window = 0

    def new_agent(self, name: str, pair: str, models_type: tuple[str] = 'A2C'):
        df = pd.read_csv(f'./data/{pair}.csv')

        training_env = MaticEnvTrain(df)
        trading_env = MaticEnvTrade(df)

        self.training_env[pair] = training_env
        self.trading_env[pair] = trading_env

        self.agents.append(Agent(name, pair, training_env, trading_env, models_type))
