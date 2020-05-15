# -*- coding: utf-8 -*-
# @Time    : 2020/2/13 16:22
# @Author  : zhangfang
import pandas as pd
from data_engine.instrument.future import Future
from data_engine.data_factory import DataFactory
import data_engine.setting as setting
import numpy as np
import math


if __name__ == "__main__":
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_LOCAL)
    client = DataFactory.get_mongo_client()
    symbols_all = ['C', 'CS', 'A', 'B', 'M', 'RM', 'Y', 'P', 'OI', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA',
                   'SC', 'FU', 'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG',
                   'SF', 'SM', 'SP', 'IF', 'IH', 'IC', 'T', 'TF', 'AG', 'AU', 'CF', 'SR', 'JD',
                   'AP', 'CJ']
    # symbols_all = ['C', 'CS', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU',
    #                    'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM', 'IF', 'IH',
    #                    'IC', 'T', 'TF', 'AG', 'AU', 'JD', 'AP', 'CJ', 'CF', 'SR']
    fold = 'G:/trading/'
    fold = 'E:/Strategy/MOMENTUM/backtest/'
    df = []
    method = '30_15'
    s_date = '2010-01-01'
    e_date = '2020-01-01'
    for symbol in symbols_all:
        result_folder = 'e://Strategy//MOMENTUM//resRepo_momentum_open_exec_%s' % (
            '_'.join([i for i in [symbol]]))
        daily_returns = pd.read_csv(result_folder + '//daily_returns.csv', header=None)
        daily_returns.columns = ['trade_date', 'daily_return']
        temp = daily_returns[
            (daily_returns['trade_date'] >= s_date) & (daily_returns['trade_date'] < e_date)]
        try:
            sharp = np.mean(temp.daily_return) / np.std(temp.daily_return) * math.pow(252, 0.5)
        except Exception as e:
            print(str(e))
            sharp = -1
        df.append([s_date, e_date, symbol, sharp])

    df = pd.DataFrame(df, columns=['s_date', 'e_date', 'symbol', 'sharp'])
    print(df)
    df.to_csv(fold + 'sharpe_state' + '_20109.csv', encoding='gbk')