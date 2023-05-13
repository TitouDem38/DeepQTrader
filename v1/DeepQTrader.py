import random
import sys

import gym
import numpy as np
import tensorflow as tf
from collections import deque

from v1.DataManager import DataManager
from v1.TradingEnv import TradingEnv

import pandas as pd
import datetime

# from memory_profiler import profile

class DDQNAgent:

    @profile
    def __init__(self, env, gamma=0.95, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
                 batch_size=64):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = 10
        self.memory = deque(maxlen=15000)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=env.observation_space.shape[0] * env.observation_space.shape[1],
                                  activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(env.action_space.n, activation='linear')
        ])

        self.target_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=env.observation_space.shape[0] * env.observation_space.shape[1],
                                    activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(env.action_space.n, activation='linear')
        ])

        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.target_model.compile(optimizer=self.optimizer, loss=self.loss_fn)

    @profile
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.model.predict(state, verbose=0)

            return np.argmax(q_values) - 2

    @profile
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @profile
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = np.zeros(
            (self.batch_size, 2 * self.env.observation_space.shape[0] * self.env.observation_space.shape[1] + 3))
        # (batch_size, size of state + action + reward + next_state + done)

        if len(self.memory) == self.batch_size:
            samples = np.array(self.memory, dtype=object)
        else:
            batch_index = np.random.randint(0, len(self.memory) - self.batch_size)
            samples = np.array(self.memory, dtype=object)[batch_index:batch_index + self.batch_size]

        # samples = random.sample(self.memory, self.batch_size)

        for i, (state, action, reward, next_state, done) in enumerate(samples):
            minibatch[i] = np.concatenate([state.flatten(), [action, reward], next_state.flatten(), [done]])

        states = minibatch[:, :self.env.observation_space.shape[0] * self.env.observation_space.shape[1]]
        actions = minibatch[:, self.env.observation_space.shape[0] * self.env.observation_space.shape[1]]
        rewards = minibatch[:, self.env.observation_space.shape[0] * self.env.observation_space.shape[1] + 1]
        next_states = minibatch[:, self.env.observation_space.shape[0] * self.env.observation_space.shape[1] + 2:-1]
        dones = minibatch[:, -1]

        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        best_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][int(actions[i] + 2)] = rewards[i]
            else:
                q_values[i][int(actions[i] + 2)] = rewards[i] + self.gamma * next_q_values[i][best_actions[i]]

        with tf.GradientTape() as tape:
            loss = self.loss_fn(q_values, self.model(states))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables), verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @profile
    def train(self, num_episodes):
        with tf.device('/device:CPU:0'):
            memory_size = sys.getsizeof(self.memory)

            for episode in range(num_episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, env.observation_space.shape[0] * env.observation_space.shape[1]])
                done = False
                total_reward = 0
                step = 0

                while not done:
                    action = self.act(state)


                    env.render(episode + 1, num_episodes, save=True)

                    next_state, reward, done, _, infos = env.step(action)

                    next_state = np.reshape(next_state,
                                            [1, env.observation_space.shape[0] * env.observation_space.shape[1]])
                    total_reward += reward
                    self.remember(state, action, reward, next_state, done)

                    state = next_state
                    self.replay()

                    if step % self.target_update_freq == 0:
                        self.target_model.set_weights(self.model.get_weights())     # DDQN

                    if sys.getsizeof(self.memory) != memory_size:
                        memory_size = sys.getsizeof(self.memory)
                        # print(f'Memory size : {memory_size / 1024 / 1024:.2f} MB')
                        # print(f'Models size : {(sys.getsizeof(self.model) + sys.getsizeof(self.target_model))/ 1024 / 1024:.2f} MB')

                        with open('v1\memory_size.txt', 'a') as f:
                            f.write(f'{episode}[{step}] - {len(self.memory)} items '
                                    f'- {memory_size / 1024 / 1024:.4f} MB - {(sys.getsizeof(self.model) + sys.getsizeof(self.target_model)) / 1024 / 1024:.4f} MB - '
                                    f'{sys.getsizeof(self) / 1024 / 1024:.4f} MB\n')

                    step += 1

                print('---------------------------------------------------------------')
                print("Episode {}/{}: \n\tTotal Reward = {}".format(episode + 1, num_episodes, total_reward))

                agent.model.save('v1\DeepQTrader.h5')

                env.render_all(episode + 1)

    def validate(self, df):
        self.env = TradingEnv(df=df, window_size=5)

        state = self.env.reset()
        state = np.reshape(state, [1, self.env.observation_space.shape[0] * self.env.observation_space.shape[1]])
        done = False
        total_reward = 0

        while not done:
            action = self.act(state)
            next_state, reward, done, _, infos = self.env.step(action)
            next_state = np.reshape(next_state,
                                    [1, self.env.observation_space.shape[0] * self.env.observation_space.shape[1]])
            total_reward += reward
            state = next_state

        print('---------------------------------------------------------------')
        print("Total Reward = {}".format(total_reward))

        self.env.render_all(1)


if __name__ == '__main__':
    # dm = DataManager(symbol='MATICUSDT')
    df = pd.read_csv('./data/MATICUSDT.csv')

    env = TradingEnv(df=df, window_size=10)

    agent = DDQNAgent(env=env)
    agent.train(num_episodes=200)

    # dm_test = DataManager(symbol='BTCUSDT')

    # agent.test(dm_test)
