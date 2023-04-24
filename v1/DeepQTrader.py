import random

import gym
import numpy as np
import tensorflow as tf
from collections import deque

from v1.DataManager import DataManager
from v1.TradingEnv import TradingEnv


class DQNAgent:
    def __init__(self, env, gamma=0.95, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
                 batch_size=64):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.memory = deque(maxlen=20000)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=env.observation_space.shape[0] * env.observation_space.shape[1], activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(env.action_space.n, activation='linear')
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.model.predict(state)

            return np.argmax(q_values) - 2

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            print('Not enough samples in memory to replay. (', len(self.memory), ' / ', self.batch_size, ')')
            return

        minibatch = np.zeros(
            (self.batch_size, 2 * self.env.observation_space.shape[0] * self.env.observation_space.shape[1] + 3))
        # (batch_size, size of state + action + reward + next_state + done)

        samples = random.sample(self.memory, self.batch_size)
        for i, (state, action, reward, next_state, done) in enumerate(samples):
            minibatch[i] = np.concatenate([state.flatten(), [action, reward], next_state.flatten(), [done]])

        states = minibatch[:, :self.env.observation_space.shape[0] * self.env.observation_space.shape[1]]
        actions = minibatch[:, self.env.observation_space.shape[0] * self.env.observation_space.shape[1]]
        rewards = minibatch[:, self.env.observation_space.shape[0] * self.env.observation_space.shape[1] + 1]
        next_states = minibatch[:, self.env.observation_space.shape[0] * self.env.observation_space.shape[1] + 2:-1]
        dones = minibatch[:, -1]

        q_values = self.model.predict(states)
        next_q_values = self.model.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][int(actions[i] + 2)] = rewards[i]
            else:
                q_values[i][int(actions[i] + 2)] = rewards[i] + self.gamma * next_q_values[i][int(actions[i] + 2)]

        with tf.GradientTape() as tape:
            loss = self.loss_fn(q_values, self.model(states))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            state = np.reshape(state, [1, env.observation_space.shape[0] * env.observation_space.shape[1]])
            done = False
            total_reward = 0
            while not done:
                action = self.act(state)

                env.render(draw=False)

                next_state, reward, done, _, infos = env.step(action)
                next_state = np.reshape(next_state, [1, env.observation_space.shape[0] * env.observation_space.shape[1]])
                total_reward += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay()

            print("Episode {}/{}: Total Reward = {}".format(episode + 1, num_episodes, total_reward))

            env.render_all(episode + 1)


if __name__ == '__main__':
    dm = DataManager()

    env = TradingEnv(df=dm.data, window_size=20)

    agent = DQNAgent(env=env)
    agent.train(num_episodes=1000)

    agent.model.save('v1\DeepQTrader.h5')
