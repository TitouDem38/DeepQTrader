import datetime

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt

import time
import tensorflow as tf

class TradingEnv(gym.Env):  # v1 - discrete environment for a unique trading pair (BTCUSDT)

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        self.df = df  # The dataframe of this trading environment
        self.window_size = window_size  # The window size of each observation

        self.seed(42)
        self.observation_shape = (self.window_size, len(self.df.columns) + 2)  # The shape of each observation

        # Spaces
        self.action_space = spaces.Discrete(5, start=-2)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float64)

        # Episode-specific variables
        self._start_step = self.window_size  # To get the first observation
        self._end_step = len(self.df) - 1  # To get the last observation

        # Environment variables
        self._current_step = None
        self._done = None

        # State variables
        self._usd_balance = None
        self._asset_balance = None
        self._total_net_worth = None
        self._actions = None
        self._rewards = None

        self._bought_price = None

        # Constants
        self.trading_cost_pct = 0.01  # 1% trading cost
        # self._hold_interval = 0.05     # Hold between -hold_interval and hold_interval (in %) - deprecated for discrete space

        self.trade_investment_percentage = 0.33  # The agent will only use 20% of the wallet for each trade
        self.min_trade_balance = 0.2  # The agent will get a penalty if the balance is below this value

        self.timestamp = None

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def reset(self):
        self._current_step = self._start_step + 1
        self._done = False

        self._usd_balance = [1 for i in range(self.window_size)]
        self._asset_balance = [0 for i in range(self.window_size)]

        self._total_net_worth = [self._usd_balance[-1]]
        self._actions = [0]
        self._rewards = [0]

        self._bought_price = 0

        self.timestamp = datetime.datetime.now().timestamp()

        return self._next_observation()

    def _next_observation(self):
        start_step = self._current_step - self.window_size

        usd_balance = np.array([self._usd_balance[-self.window_size + i] for i in range(self.window_size)]).reshape(-1, 1)
        asset_balance = np.array([self._asset_balance[-self.window_size + i] for i in range(self.window_size)]).reshape(-1, 1)

        # concatenate data from df and balances into obs
        obs = np.concatenate((self.df.iloc[start_step:self._current_step].values, usd_balance, asset_balance), axis=1)

        return obs

    def step(self, action):
        assert self.action_space.contains(action)

        current_price = self.df.iloc[self._current_step]['close']

        match action:
            case 1:
                # Spend trade_investment_percentage % of usd balance to buy assets
                nb_assets_bought = self._usd_balance[-1] * self.trade_investment_percentage / \
                                      (current_price * (1 + self.trading_cost_pct))

                self._usd_balance.append(self._usd_balance[-1] * (1 - self.trade_investment_percentage))
                self._asset_balance.append(self._asset_balance[-1] + nb_assets_bought)
                self._bought_price = current_price

            case 2:
                # Spend 2*trade_investment_percentage % of usd balance to buy assets
                nb_assets_bought = self._usd_balance[-1] * 2 * self.trade_investment_percentage / \
                                   (current_price * (1 + self.trading_cost_pct))

                self._usd_balance.append(self._usd_balance[-1] * (1 - 2 * self.trade_investment_percentage))
                self._asset_balance.append(self._asset_balance[-1] + nb_assets_bought)
                self._bought_price = current_price

            case -1:
                # Sell trade_investment_percentage % of asset balance
                nb_assets_sold = self._asset_balance[-1] * self.trade_investment_percentage

                self._asset_balance.append(self._asset_balance[-1] - nb_assets_sold)
                self._usd_balance.append(self._usd_balance[-1] + nb_assets_sold * (1 - self.trading_cost_pct) * current_price)

            case -2:
                # Sell 2*trade_investment_percentage % of asset balance
                nb_assets_sold = self._asset_balance[-1] * 2 * self.trade_investment_percentage

                self._asset_balance.append(self._asset_balance[-1] - nb_assets_sold)
                self._usd_balance.append(self._usd_balance[-1] + nb_assets_sold * (1 - self.trading_cost_pct) * current_price)

            case 0:
                # Hold
                self._usd_balance.append(self._usd_balance[-1])
                self._asset_balance.append(self._asset_balance[-1])

        # Compute reward
        reward = 1 * (self._usd_balance[-1] + self._asset_balance[-1] * current_price - self._total_net_worth[-1])

        self._actions.append(action)

        # Update current step
        self._current_step += 1

        # Update done flag
        self._done = self._current_step > self._end_step

        # Compute reward
        self._rewards.append(reward)
        # Reward is the difference between the current net worth and the previous net worth

        # Update total net worth
        self._total_net_worth.append(self._usd_balance[-1] + self._asset_balance[-1] * current_price)

        # Compute next observation
        obs = self._next_observation()

        return obs, self._rewards[-1], self._done, False, {}

    def render(self, episode, num_episodes, render_every=150, save=False):
        if (self._current_step - self._start_step) % render_every == 0:

            # Compute remaining time
            delta_t = datetime.datetime.timestamp(datetime.datetime.now()) - self.timestamp
            self.timestamp = datetime.datetime.timestamp(datetime.datetime.now())

            remaining_steps = self._end_step - self._current_step + (self._end_step - self._start_step) * (
                        num_episodes - episode - 1)

            total_steps_per_episode = self._end_step - self._start_step
            total_steps = total_steps_per_episode * num_episodes

            remaining_time = datetime.timedelta(seconds=delta_t * remaining_steps / render_every)

            # Print step, balances, action, reward on a single line
            actions = {2: 'Buy 66%', 1: 'Buy 33%', 0: 'Hold', -1: 'Sell 33%', -2: 'Sell 66%'}
            rep_actions = [f'{actions[i]} : {self._actions.count(i) / len(self._actions) * 100:.1f}% | '
                           for i in range(-2, 3)]

            print(f'Step: {self._current_step - self._start_step} / {self._end_step - self._start_step}'
                  f' - Episode {(self._current_step - self._start_step) * 100 / total_steps_per_episode:.2f}% - '
                  f'Training {100 * (1 - remaining_steps/total_steps):.2f}% - Remaining time : {str(remaining_time)}\n'
                  f'\tBalances - USD: {self._usd_balance[-1]:.2f} $ | '
                  f'Asset: {self._asset_balance[-1]:.2f} BTC | '
                  f'Net worth: {self._total_net_worth[-1]:.2f} $ \n'
                  f'\tActions: {"".join(rep_actions)} \n'
                  f'\tTotal reward: {sum(self._rewards):.2f}\n')

            if save:
                with open(f'./training_logs/episode_{episode}.csv', 'a') as f:
                    for i in range(render_every):
                        f.write(f'{self._current_step - self._start_step - i},'
                                f'{self._usd_balance[-1 - i]:.2f},'
                                f'{self._asset_balance[-1 - i]:.2f},'
                                f'{self._total_net_worth[-1 - i]:.2f},'
                                f'{self._actions[-1 - i]},'
                                f'{self._rewards[-1 - i]:.2f},'
                                f'{sum(self._rewards[:-1 -i]):.2f}\n')

    def render_all(self, episode):
        # Render the whole episode

        fig, ax1 = plt.subplots()
        ax1.set_xlim([self._start_step, self._end_step])
        ax1.set_ylim([-.1, 1.1])

        ax2 = ax1.twinx()

        steps = np.arange(self._start_step, self._end_step + 1)

        ax2.plot(steps, self._total_net_worth)
        ax1.plot(steps, self.df.iloc[self._start_step:self._end_step + 1]['close'], 'k:')

        ax1.plot(steps, self._rewards, 'r--')

        # Hold strategy comparison
        max_assets = self._total_net_worth[0] / self.df.iloc[self._start_step]['close']
        final_balance = max_assets * self.df.iloc[self._end_step]['close']

        if max([final_balance, max(self._total_net_worth)]) > self._total_net_worth[0] / self.df.iloc[self._start_step]['close']:
            ax2.set_ylim([-.1, max([final_balance, max(self._total_net_worth)]) * 1.1])
            ax1.set_ylim([-.1, max([final_balance, max(self._total_net_worth)]) * self.df.iloc[self._start_step]['close'] / self._total_net_worth[0] * 1.1])
        else:
            ax2.set_ylim([-.1, self._total_net_worth[0] / self.df.iloc[self._start_step]['close'] * 1.1])

        for i, action in enumerate(self._actions):
            marker = 'o'

            if type(action) == int:
                marker = 'x'

            match action:
                case -2:
                    ax1.plot(self._start_step + i, self.df.iloc[self._start_step + i]['close'], f'r{marker}', markersize=4)
                case -1:
                    ax1.plot(self._start_step + i, self.df.iloc[self._start_step + i]['close'], f'r{marker}', markersize=3)
                case 0:
                    pass
                case 1:
                    ax1.plot(self._start_step + i, self.df.iloc[self._start_step + i]['close'], f'g{marker}', markersize=4)
                case 2:
                    ax1.plot(self._start_step + i, self.df.iloc[self._start_step + i]['close'], f'g{marker}', markersize=3)

        plt.title(f'Total reward: {sum(self._rewards):.2f}')
        plt.savefig(f'plots/episode {episode} - {int(sum(self._rewards))}.png')
        plt.close()

        actions = {2: 'Buy 66%', 1: 'Buy 33%', 0: 'Hold', -1: 'Sell 33%', -2: 'Sell 66%'}
        rep_actions = [f'{actions[i]} : {self._actions.count(i) / len(self._actions) * 100:.1f}% | '
                       for i in range(-2, 3)]

        print(f'\t Actions: {"".join(rep_actions)}')
        print('---------------------------------------------------------------\n')
