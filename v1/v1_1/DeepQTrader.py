import random
import sys

import numpy as np
import tensorflow as tf
from collections import deque

from keras.regularizers import l2

from v1.v1_1.TradingEnv import TradingEnv

import pandas as pd

class DDQNAgent:

    def __init__(self, env, gamma=0.9, learning_rate=0.001, epsilon=1.0, eps_decay=0.9995, epsilon_min=0.01,
                 batch_size=1024, memory_capacity=50000, architecture=(256, 256)):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = 0.01
        self.eps_exp_decay = eps_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = 100
        self.l2_reg = 0.0001

        self.training = True

        self.memory = deque(maxlen=memory_capacity)
        self.idx = tf.range(self.batch_size)

        self.losses = [0]

        self.network_architecture = architecture

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.model = self.build_model(trainable=True)
        self.target_model = self.build_model(trainable=False)

        self.update_target()

        self.episodes = 0

    def build_model(self, trainable=False):
        layers = []

        for i, nodes in enumerate(self.network_architecture):
            if i == 0:
                layers.append(tf.keras.layers.Dense(units=nodes,
                                                    input_dim=self.env.observation_space.shape[0] * self.env.observation_space.shape[1] if i == 0 else None,
                                                    activation='relu',
                                                    kernel_regularizer=l2(self.l2_reg),
                                                    trainable=trainable,
                                                    name='dense_' + str(i)))

        layers.append(tf.keras.layers.Dropout(0.1))
        layers.append(tf.keras.layers.Dense(units=self.env.action_space.n,
                                            activation='linear',
                                            trainable=trainable,
                                            name='output'))

        model = tf.keras.Sequential(layers)
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def eps_greedy_policy(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.model.predict(state, verbose=0)

            return np.argmax(q_values) - 2

    def remember(self, state, action, reward, next_state, not_done):
        if not_done:
            pass
        else:
            if self.training:
                if self.epsilon > self.epsilon_min:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.eps_exp_decay

        self.memory.append((state, action, reward, next_state, not_done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = map(np.array, zip(*random.sample(self.memory, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        states = states.squeeze()
        next_states = next_states.squeeze()

        next_q_values = self.model.predict_on_batch(next_states)
        best_actions = np.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_model.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + self.gamma * target_q_values * not_done

        q_values = self.model.predict_on_batch(states)

        for i in range(self.batch_size):
            q_values[i][actions[i]] = targets[i]

        loss = self.model.train_on_batch(states, q_values)
        self.losses.append(loss)

    def train(self, num_episodes):
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            print('Using GPU')
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        else:
            print('Using CPU')

        memory_size = sys.getsizeof(self.memory)

        self.training = True

        for episode in range(num_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, env.observation_space.shape[0] * env.observation_space.shape[1]])
            done = False
            total_reward = 0
            step = 0

            while not done:
                action = self.eps_greedy_policy(state)

                env.render(episode + 1, num_episodes, save=False)

                next_state, reward, done, _, infos = env.step(action)
                next_state = np.reshape(next_state,
                                        [1, env.observation_space.shape[0] * env.observation_space.shape[1]])
                total_reward += reward

                self.remember(state, action, reward, next_state, 0 if done else 1)

                state = next_state
                self.replay()

                if step % self.target_update_freq == 0:
                    self.update_target()     # DDQN

                step += 1

            print('---------------------------------------------------------------')
            print(f"Episode {episode + 1}/{num_episodes}: \n\tLoss = {self.losses[-1]} \n\t Epsilon = {self.epsilon:.2f}\n\tTotal Reward = {total_reward}")

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

    env = TradingEnv(df=df, window_size=20)

    agent = DDQNAgent(env=env)

    agent.train(num_episodes=200)

    # dm_test = DataManager(symbol='BTCUSDT')

    # agent.test(dm_test)
