import os
import time
import tejapi
import talib as ta
from talib import abstract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime
from datetime import timedelta
import configparser
import re
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

class ML_stock:
    def __init__(self, config_file='config.ini') -> None:
        self.config_file = config_file
        self.api_keys = self.get_api_keys(self.config_file)
        self.ini()

    def get_api_keys(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        api_keys = {}
        if 'API_KEYS' in config:
            for key, value in config['API_KEYS'].items():
                api_keys[key] = value

        return api_keys
    
    def ini(self):
        os.environ['TEJAPI_KEY'] = self.api_keys.get('api_key')
        os.environ['TEJAPI_BASE'] = self.api_keys.get('api_base')

    def get_fundamental(self, start, end, tickers, column):
        import TejToolAPI
        from zipline.sources.TEJ_Api_Data import get_universe
        start_dt, end_dt = pd.Timestamp(start, tz='utc'), pd.Timestamp(end, tz='utc')
        df = TejToolAPI.get_history_data(start = start_dt,
                                        end = end_dt,
                                        ticker = tickers,
                                        columns = column,
                                        transfer_to_chinese = False)
        
        mask = df['Close'] > 10
        df = df[mask]
        return df
  
    def calculate_all_technical_indicators(self, df):
        df_all = df.copy()
        change = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume_1000_Shares': 'volume'
        }
        df_all.rename(columns=change, inplace=True)
        df_all['MOM'] = abstract.MOM(df_all['close'])
        df_all['RSI'] = abstract.RSI(df_all['close'])

        return df_all
    
    def preprocessing(self, df):
        df_all = df.copy()
        data = df_all.groupby('coid').apply(self.calculate_all_technical_indicators)
        data.reset_index(drop=True, inplace=True)
        data['coid'] = pd.to_numeric(data['coid'])
        
        # deal with the NAs
        data = data[~(data['Return_Rate_on_Equity_A_percent_A'].isna())].reset_index(drop=True)
        aa = data[data.isnull().any(axis=1)]
        nan_col = aa.columns[aa.isnull().any()].tolist()
        for col in nan_col:
            aa[col] = aa.groupby('coid')[col].transform(lambda x: x.fillna(x.mean()))
        if data.isnull().any().any():
            for col in nan_col:
                data[col] = data[col].fillna(0)
        data = data.drop(columns = ['Return_Rate_on_Equity_A_percent_A', 'Return_Rate_on_Equity_A_percent_TTM'])
        data = data.sort_values(by=['coid', 'mdate']).reset_index(drop=True)

        return data
