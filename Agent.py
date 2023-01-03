import numpy as np
import pandas
import pandas as pd
import gym
import time

# RL models
from stable_baselines3 import PPO  # will be used for bearish market
from stable_baselines3 import A2C  # will be used for bullish market

# environments
from gym_env.EnvTrain import MaticEnvTrain
from gym_env.EnvTrade import MaticEnvTrade


class Agent:
    MODEL_FILES_DIRECTORY = './models/'

    def __init__(self, name: str, pair: str, train_env: gym.Env, trade_env: gym.Env, models_type: tuple[str] = 'A2C'):
        self.name = name
        self.pair = pair

        self.train_env = train_env
        self.trade_env = trade_env

        self.models_type = models_type

    def train(self, model_type, time_steps=25000):

        start_time = time.time()

        match model_type:
            case 'A2C':
                model = A2C('MlpPolicy', self.train_env)

            case 'PPO2':
                model = PPO('MlpPolicy', self.train_env, ent_coef=0.005)

            case other:
                return None

        model.learn(total_timesteps=time_steps)
        end_time = time.time()

        model.save(f'{self.MODEL_FILES_DIRECTORY}{model_type}')

        print(f'{model_type} training done in {(end_time - start_time) / 60} minutes')

        return model

    def compute_next_trade(self):
        start_time = time.time()

    def run_trading(self, df: pandas.DataFrame, retrain_window: int, validation_window: int) -> None:
        """
        Compute the trading actions for a given period using the best model available

        :param df: Main dataframe
        :param retrain_window: Number of training periods to retrain the model before
        :param validation_window: Number of training periods to validate the best model before
        """
