import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt


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

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._current_step = self._start_step + 1
        self._done = False

        self._usd_balance = [1 for i in range(self.window_size)]
        self._asset_balance = [0 for i in range(self.window_size)]

        self._total_net_worth = [self._usd_balance[-1]]
        self._actions = [0]
        self._rewards = [0]

        self._bought_price = 0

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

                if self._usd_balance[-1] < self.min_trade_balance:
                    reward = -10
                else:
                    reward = self._usd_balance[-1] + self._asset_balance[-1] * current_price - self._total_net_worth[-1]

            case 2:
                # Spend 2*trade_investment_percentage % of usd balance to buy assets
                nb_assets_bought = self._usd_balance[-1] * 2 * self.trade_investment_percentage / \
                                   (current_price * (1 + self.trading_cost_pct))

                self._usd_balance.append(self._usd_balance[-1] * (1 - 2 * self.trade_investment_percentage))
                self._asset_balance.append(self._asset_balance[-1] + nb_assets_bought)
                self._bought_price = current_price

                if self._usd_balance[-1] < self.min_trade_balance:
                    reward = -10
                else:
                    reward = self._usd_balance[-1] + self._asset_balance[-1] * current_price - self._total_net_worth[-1]

            case -1:
                # Sell trade_investment_percentage % of asset balance
                nb_assets_sold = self._asset_balance[-1] * self.trade_investment_percentage

                self._asset_balance.append(self._asset_balance[-1] - nb_assets_sold)
                self._usd_balance.append(self._usd_balance[-1] + nb_assets_sold * (1 - self.trading_cost_pct) * current_price)

                if self._asset_balance[-1] < self.min_trade_balance:
                    reward = -10
                else:
                    reward = self._usd_balance[-1] + self._asset_balance[-1] * current_price - self._total_net_worth[-1]

            case -2:
                # Sell 2*trade_investment_percentage % of asset balance
                nb_assets_sold = self._asset_balance[-1] * 2 * self.trade_investment_percentage

                self._asset_balance.append(self._asset_balance[-1] - nb_assets_sold)
                self._usd_balance.append(self._usd_balance[-1] + nb_assets_sold * (1 - self.trading_cost_pct) * current_price)

                if self._asset_balance[-1] < self.min_trade_balance:
                    reward = -10
                else:
                    reward = self._usd_balance[-1] + self._asset_balance[-1] * current_price - self._total_net_worth[-1]

            case 0:
                # Hold
                self._usd_balance.append(self._usd_balance[-1])
                self._asset_balance.append(self._asset_balance[-1])

                if self._bought_price == 0:
                    reward = -10
                else:
                    reward = (self._usd_balance[-1] + self._asset_balance[-1] * current_price - self._total_net_worth[-1]) / self._bought_price

                if reward > 0:
                    reward *= 20

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

    def render(self, draw=True):
        action = self._actions[-1]

        # Print step, balances, action, reward on a single line
        actions = {2: 'Buy 66%', 1: 'Buy 33%', 0: 'Hold', -1: 'Sell 33%', -2: 'Sell 66%'}

        print(f'\rStep: {self._current_step} | '
              f'USD: {self._usd_balance[-1]:.2f} | '
              f'Asset: {self._asset_balance[-1]:.2f} | '
              f'Net worth: {self._total_net_worth[-1]:.2f} | '
              f'Action: {actions[action]} | '
              f'Reward: {self._rewards[-1]:.2f}')

    def render_all(self, episode):
        # Render the whole episode

        fig, ax1 = plt.subplots()
        ax1.set_xlim([self._start_step, self._end_step])
        ax1.set_ylim([0, 1.1])

        ax2 = ax1.twinx()

        steps = np.arange(self._start_step, self._end_step + 1)

        ax2.plot(steps, self._total_net_worth)
        ax1.plot(steps, self.df.iloc[self._start_step:self._end_step + 1]['close'], 'k:')

        # Hold strategy comparison
        max_assets = self._total_net_worth[0] / self.df.iloc[self._start_step]['close']
        final_balance = max_assets * self.df.iloc[self._end_step]['close']

        if max([final_balance, max(self._total_net_worth)]) > self._total_net_worth[0] / self.df.iloc[self._start_step]['close']:
            ax2.set_ylim([0, max([final_balance, max(self._total_net_worth)]) * 1.1])
            ax1.set_ylim([0, max([final_balance, max(self._total_net_worth)]) * self.df.iloc[self._start_step]['close'] / self._total_net_worth[0] * 1.1])
        else:
            ax2.set_ylim([0, self._total_net_worth[0] / self.df.iloc[self._start_step]['close'] * 1.1])

        for i, action in enumerate(self._actions):
            match action:
                case -2:
                    ax1.plot(self._start_step + i, self.df.iloc[self._start_step + i]['close'], 'ro', markersize=4)
                case -1:
                    ax1.plot(self._start_step + i, self.df.iloc[self._start_step + i]['close'], 'ro', markersize=3)
                case 0:
                    pass
                case 1:
                    ax1.plot(self._start_step + i, self.df.iloc[self._start_step + i]['close'], 'go', markersize=4)
                case 2:
                    ax1.plot(self._start_step + i, self.df.iloc[self._start_step + i]['close'], 'go', markersize=3)

        plt.title(f'Total reward: {sum(self._rewards):.2f}')
        plt.savefig(f'plots/episode {episode} - {int(sum(self._rewards))}.png')
        plt.close()
