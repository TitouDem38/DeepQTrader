import pandas as pd

from typing import Union, Tuple
from binance.client import Client
from finta import TA

from sklearn.decomposition import PCA

class DataManager:

    def __init__(self, symbol) -> None:
        self._client = Client("clef_api", "clef_secrete", testnet=False)
        self._symbol: str | None = symbol

        # Gestion dataframe
        self._dataframe: pd.DataFrame | None = None
        self._columns_names: list[str] = ['time', 'open', 'high', 'low', 'close', 'volume']

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe.copy()

    def get_historical_klines(self,
                              start_stop_day: Union[int, Tuple[int, int]] = (1, 31),
                              start_stop_month: Union[str, Tuple[str, str]] = "Jan",
                              start_stop_year: Union[int, Tuple[int, int]] = 2021,
                              kline_interval: str = "1m",
                              format_df: bool = False) -> None:
        """
        *kline_interval* constants : https://python-binance.readthedocs.io/en/latest/constants.html

        .. warning::
            if you take only one day, do not put the last day of a month
        |
        klines forms:
        .. code-block::
            [
              [
                1499040000000,      // Open time
                "0.01634790",       // Open
                "0.80000000",       // High
                "0.01575800",       // Low
                "0.01577100",       // Close
                "148976.11427815",  // Volume
                1499644799999,      // Close time
                "2434.19055334",    // Quote asset volume
                308,                // Number of trades
                "1756.87402397",    // Taker buy base asset volume
                "28.46694368",      // Taker buy quote asset volume
                "17928899.62484339" // Ignore.
              ]
            ]
        :param format_df:
        :param start_stop_day: Day or tuple of days
        :type start_stop_day: Union[int, Tuple[int, int]]
        :param start_stop_month: Month or tuple of months
        :type start_stop_month: Union[str, Tuple[str, str]]
        :param start_stop_year: Year or tuple of years
        :type start_stop_year: Union[int, Tuple[int, int]]
        :param kline_interval: Interval of time
        :type kline_interval: Union[int, Tuple[int, int]]
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
            klines = self._client.get_historical_klines(self._symbol.upper(),
                                                        kline_interval,
                                                        start,
                                                        stop)
        except AttributeError:
            # Fermeture de la méthode
            print('Erreur lors de la récupération des bougies ')
            return

        # Build dataframe
        self._dataframe = pd.DataFrame(klines).iloc[:, :len(self._columns_names)]
        self._dataframe.columns = self._columns_names
        self._dataframe = self._dataframe.apply(pd.to_numeric)

        # Format dataframe
        if format_df:
            self._dataframe = self._format_dataframe(self._dataframe)

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format dataframe
        :param df:
        :return:
        """
        # Convert to datetime
        df['time'] = pd.to_datetime(df['time'], unit='ms')

        df['rsi14'] = TA.RSI(df, 14)
        df['macd'] = TA.MACD(df).MACD
        df['cci20'] = TA.CCI(df, 20)
        df['sma10'] = TA.SMA(df, 10)

        new_df = df[['time', 'close', 'volume', 'rsi14', 'macd', 'cci20', 'sma10']]

        new_df = new_df.dropna()
        new_df = new_df.reset_index(drop=True)

        pca = PCA(.99)
        pca.fit(new_df.iloc[:, 3:])

        pca_df = pca.transform(new_df.iloc[:, 3:])
        pca_df = pd.DataFrame(pca_df)

        new_df = [new_df, pca_df]
        print(new_df)

        return new_df


if __name__ == '__main__':
    data = DataManager("MATICBUSD")
    data.get_historical_klines(kline_interval="1d", format_df=True,
                               start_stop_day=(1, 20),
                               start_stop_month=("Jan", "Dec"),
                               start_stop_year=2022)

    df = data.dataframe


