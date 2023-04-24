import numpy as np
from binance.client import Client
from typing import Union, Tuple
from sklearn.preprocessing import minmax_scale

import os
import pandas as pd
from finta import TA

import matplotlib.pyplot as plt


class DataManager:
    """
    Data manager for TradingEnv

    Loads/Downloads data from ?
    Provide data for each new episode

    Mode live for live trading after training and testing ?
    """

    def __init__(self, trading_ticks=100, symbol='BTCUSDT', normalize=True, mode='train'):
        self.trading_ticks = trading_ticks
        self.symbol = symbol
        self.normalize = normalize
        self.mode = mode

        self.binance_client = Client(os.environ['API_KEY'], os.environ['API_KEY_SECRET'], testnet=False)

        self.data = self.load_data()

        self.min_values = self.data.min(axis=0)
        self.max_values = self.data.max(axis=0)
        self.step = 0
        self.offset = None

    def load_data(self):
        """
        Load data from file or download it from ?

        :return: DataFrame
        """
        print(f'Loading data for {self.symbol}...')

        if self.mode == 'live':
            ...
            data = None

        else:
            df = self.get_historical_klines(start_stop_day=(1, 24),
                                            start_stop_month=('Jan', 'Mar'),
                                            start_stop_year=(2022, 2023),
                                            kline_interval='1d')

            data = self.preprocess_data(df)

        return data

    def get_historical_klines(self,
                              start_stop_day: Union[int, Tuple[int, int]] = (1, 31),
                              start_stop_month: Union[str, Tuple[str, str]] = "Mar",
                              start_stop_year: Union[int, Tuple[int, int]] = 2023,
                              kline_interval: str = "1h") -> list:
        """
        *kline_interval* constants : https://python-binance.readthedocs.io/en/latest/constants.html
        """

        # Time data
        start: str = ""
        stop: str = ""

        # Day
        if type(start_stop_day) is int:
            start += f"{start_stop_day}"
            stop += f"{start_stop_day + 1}"
        else:
            start += f"{start_stop_day[0]}"
            stop += f"{start_stop_day[1]}"

        # Month
        if type(start_stop_month) is str:
            start += f" {start_stop_month}"
            stop += f" {start_stop_month}"
        else:
            start += f" {start_stop_month[0]}"
            stop += f" {start_stop_month[1]}"

        # Year
        if type(start_stop_year) is int:
            start += f" {start_stop_year}"
            stop += f" {start_stop_year}"
        else:
            start += f" {start_stop_year[0]}"
            stop += f" {start_stop_year[1]}"

        # Get data
        try:
            klines = self.binance_client.get_historical_klines(self.symbol.upper(), kline_interval, start, stop)
        except AttributeError:
            print('Error: symbol not found')
            return None

        return klines

    def preprocess_data(self, klines):
        # Build dataframe
        columns_names = ['time', 'open', 'high', 'low', 'close', 'volume']

        df = pd.DataFrame(klines).iloc[:, :len(columns_names)]
        df.columns = columns_names
        df = df.apply(pd.to_numeric)

        # Compute technical indicators

        # Compute RSIs
        df['rsi7'] = TA.RSI(df, 7)
        df['rsi14'] = TA.RSI(df, 14)
        df['rsi21'] = TA.RSI(df, 21)

        # Compute MACD
        df['macd'] = TA.MACD(df).MACD

        # Compute MAs
        df['ma7'] = TA.SMA(df, 7)
        df['ma14'] = TA.SMA(df, 14)
        df['ma21'] = TA.SMA(df, 21)
        df['ma50'] = TA.SMA(df, 50)

        # Compute CCI
        df['cci'] = TA.CCI(df, 5)

        df = df.replace((np.inf, -np.inf), np.nan).dropna()

        df = df.drop(columns=['time'])

        if self.normalize:
            df = pd.DataFrame(df / df.max(axis=0),
                              columns=df.columns,
                              index=df.index)

        print(f'Data loaded for {self.symbol} ({len(df)} rows)')

        return df

    def reset(self):
        """Provides starting index for time series and resets step"""
        self.step = 0
        self.offset = np.random.randint(0, len(self.data) - self.trading_ticks)
        return self.data.iloc[self.offset:self.offset + self.trading_ticks]

    def take_step(self):
        """Returns data for current trading tick and done signal"""
        obs = self.data.iloc[self.offset + self.step].values    # current observation at t = offset + step
        self.step += 1
        done = self.step > self.trading_ticks
        return obs, done

def plot_image(df):
    # convert a dataframe into an image : each pixel is (r, g, b) value corresponding to (close, rsi7, cci) price
    # the format of the image is (int(sqrt(len(df))), int(sqrt(len(df))), 3)

    # normalize data
    df = df.copy()
    df['close'] = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
    df['rsi7'] = (df['rsi7'] - df['rsi7'].min()) / (df['rsi7'].max() - df['rsi7'].min())
    df['cci'] = (df['cci'] - df['cci'].min()) / (df['cci'].max() - df['cci'].min())

    # convert to image
    img = np.zeros((int(np.sqrt(len(df))), int(np.sqrt(len(df))), 3))
    for i in range((int(np.sqrt(len(df)))**2)):
        img[i // int(np.sqrt(len(df))), i % int(np.sqrt(len(df))), 0] = df.iloc[i]['close']
        img[i // int(np.sqrt(len(df))), i % int(np.sqrt(len(df))), 1] = df.iloc[i]['rsi7']
        img[i // int(np.sqrt(len(df))), i % int(np.sqrt(len(df))), 2] = df.iloc[i]['cci']

    # plot
    plt.imshow(img)

    close = np.zeros(np.size(img))
    print(np.size(close))

    rsi7 = np.zeros(np.size(img))
    cci = np.zeros(np.size(img))

    plt.figure()
    plt.imshow(close)
    plt.figure()
    plt.imshow(rsi7)
    plt.figure()
    plt.imshow(cci)

    plt.show()

def plot_curves(df):
    # one figure for each type of indicator

    # close
    plt.figure()
    plt.plot(df['close'])
    plt.title('Close')

    # rsi7
    plt.figure()
    plt.plot(df['rsi7'])
    plt.title('RSI7')

    # cci
    plt.figure()
    plt.plot(df['cci'])
    plt.title('CCI')

    plt.show()


if __name__ == '__main__':
    dm = DataManager()

    plot_curves(dm.data)