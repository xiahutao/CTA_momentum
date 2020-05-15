# -*- coding: utf-8 -*-
"""
Created on Fri Dec  13 16:34:53 2019

@author: zhangfang
"""

import pandas as pd
import numpy  as np
import datetime
from common.file_saver import file_saver
from common.decorator import runing_time
from strategy.strategy import Strategy
from data_engine.global_variable import ASSETTYPE_FUTURE,DATASOURCE_REMOTE,DATASOURCE_LOCAL
from data_engine.data_factory import DataFactory
import talib
import math


def trensferScta(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


class CtaMomentumStrategy(Strategy):
    _strategy_name = 'CtaMomentumStrategy'
    _strategy_type = 'intraday_symbols'

    def __init__(self, asset_type, symbols_list, freq, s_period_lst, l_period_lst, SW, PW, targetVol,maxLeverage,
                 period, result_fold, **kwargs):

        super(CtaMomentumStrategy, self).__init__(period=period,
                                             s_period_lst=s_period_lst,
                                             l_period_lst=l_period_lst,
                                             SW=SW,
                                             PW=PW,
                                             targetVol=targetVol,
                                             maxLeverage=maxLeverage,
                                             )
        self._freq = freq
        self._asset_type = asset_type
        self._symbol_pair_list = symbols_list

        self._symbols = set()
        for s1 in symbols_list:
            if s1 not in self._symbols:
                self._symbols.add(s1)

        self._contract_size_dict = {}
        self._tick_size_dict = {}

        self._market_data = None
        self.result_fold = result_fold

    def _get_market_info(self):
        print('================', self._symbols)
        self._contract_size_dict = DataFactory.get_contract_size_dict(symbols=list(self._symbols),
                                                                      asset_type=ASSETTYPE_FUTURE)
        self._tick_size_dict = DataFactory.get_tick_size_dict(symbols=list(self._symbols), asset_type=ASSETTYPE_FUTURE)

    def _get_history(self, startDate, endDate):
        self._market_data = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq=self._freq,
                                                          symbols=list(self._symbols), start_date=startDate,
                                                          end_date=endDate)

    def _get_history_daily(self, startDate, endDate):
        self._market_data_daily = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq='1d',
                                                                symbols=list(self._symbols), start_date=startDate,
                                                                end_date=endDate)


    def gen_signal(self, symbol, **kwargs):
        capital = kwargs['capital']
        market_data_dict = kwargs['market_data_dict']
        pos_df_list = []
        contract_size_list = self._contract_size_dict[symbol]
        print(contract_size_list)
        data = market_data_dict[symbol]
        # data.to_csv('E://data//MOMENTUM//' + symbol + '_data.csv')
        _signal_lst = []
        _signal = 0
        net = 1
        capital_intial = capital / len(self._symbol_pair_list) / self._contract_size_dict[symbol]
        data[symbol] = np.round(data['Scta'] * capital_intial / data['close'])
        signal = data[symbol]
        # signal.to_csv('E://Strategy//OCM//signal_0.csv')
        target_pos_dict = {}
        target_pos_tmp = signal
        target_pos_tmp.fillna(method='pad', inplace=True)
        target_pos_tmp.name = symbol
        target_pos_dict[symbol] = target_pos_tmp
        # target_pos_tmp.to_csv('E://Strategy//OCM//' + symbol + '_target_pos.csv')
        # if self.result_fold is not None:
        #     file_saver().save_file(target_pos_tmp, self.result_fold + '\\' + symbol + '_target_pos.csv')

        # 换月处理
        contract_id_series = data['contract_id']
        contract_id_series.name = symbol + '_contract_id'
        target_pos_df = pd.concat([target_pos_tmp, contract_id_series], axis=1)
        # target_pos_df.loc[
        #     target_pos_df[symbol + '_contract_id'] != target_pos_df[symbol + '_contract_id'].shift(-2), symbol] = 0
        target_pos_dict[symbol] = target_pos_df[symbol]

        pos_serires = target_pos_dict[symbol].copy()
        pos_serires.name = 'position'
        pos_df = pd.DataFrame(pos_serires, index=pos_serires.index)
        pos_df = pos_df.join(contract_id_series)
        pos_df['symbol'] = symbol
        pos_df['asset_type'] = self._asset_type
        pos_df['contract_size'] = self._contract_size_dict[symbol]
        pos_df['tick_size'] = self._tick_size_dict[symbol]
        pos_df['margin_ratio'] = 0.1
        pos_df['freq'] = self._freq
        pos_df['remark'] = '.'.join([self._strategy_name, self._strategy_type, symbol])

        pos_df_list.append(pos_df)
        signal_dataframe = None
        if len(pos_df_list) > 0:
            signal_dataframe = pd.concat(pos_df_list)

        return signal_dataframe

    def vol_estimator_garch(self, data_df, st=25, lt=252 * 3):  # 250*5):
        st_span = st  # min(st,len(data_df))
        lt_span = lt  # min(lt,len(data_df))
        # print(st_span, lt_span,st)
        st_vol = data_df.ewm(span=st_span, ignore_na=True, min_periods=st, adjust=False).std(bias=True)
        lt_vol = data_df.ewm(span=lt_span, ignore_na=True, min_periods=st, adjust=False).std(bias=True)
        decay_rate = 0.8
        vol = st_vol * decay_rate + lt_vol * (1 - decay_rate)
        # vol=self.cap_vol_by_rolling(vol)
        return vol

    @staticmethod
    def get_resp_curve(x, method):
        resp_curve = pd.DataFrame()
        if method == 'gaussian':
            resp_curve = np.exp(-(x ** 2) / 4.0)
        return resp_curve

    def cap_vol_by_rolling(self, vol):
        idxs = vol.index
        for idx in range(len(idxs)):
            curDate = idxs[idx]
            vol[curDate] = max(vol[curDate], 0.1)

        return vol

    @runing_time
    def _format_data(self, symbol):
        SK = self._params['s_period_lst']
        LK = self._params['l_period_lst']
        targetVol = self._params['targetVol']
        maxLeverage = self._params['maxLeverage']
        print(symbol)
        data_daily = self._market_data_daily[symbol]
        print(data_daily)
        data_daily = data_daily[data_daily['volume'] > 0][['high', 'close', 'open', 'low', 'volume', 'price_return', 'contract_id']]
        data_daily.price_return = data_daily.price_return.fillna(value=0)
        for k in range(len(SK)):
            volAdjRet = data_daily['price_return'] / data_daily['price_return'].ewm(span=SK[k], min_periods=SK[k], adjust=False).std(bias=True)
            # volAdjRet=volAdjRet.clip(-4.0,4.0)
            px_df = np.cumsum(volAdjRet)
            sig = px_df.ewm(span=SK[k], min_periods=SK[k]).mean() - px_df.ewm(span=LK[k], min_periods=SK[k]).mean()
            sig_normalized = sig / self.vol_estimator_garch(sig, 25)
            # yk_std = talib.STDDEV(yk, timeperiod=SW, nbdev=1)
            sig_resp = self.get_resp_curve(sig_normalized, 'gaussian')
            os_norm = 1.0 / 0.89
            sig = sig_normalized * sig_resp * os_norm
            realizedVol = data_daily['price_return'].ewm(span=20, ignore_na=True, min_periods=20,
                                                           adjust=False).std(bias=True) * (252 ** 0.5)
            if symbol not in ['T_VOL', 'TF_VOL']:
                realizedVol = self.cap_vol_by_rolling(realizedVol)
            riskScaler = targetVol / realizedVol
            scaledSig = riskScaler * sig
            data_daily['uk' + str(k)] = scaledSig
            data_daily['uk' + str(k)].fillna(method='pad', inplace=True)
        scaledSig = data_daily['uk0']
        leverageRaw = abs(scaledSig)
        leverageCapped = leverageRaw.apply(lambda x: min(x, maxLeverage))
        signOfSignal = scaledSig.apply(lambda x: np.sign(x))
        signalFinal = signOfSignal * leverageCapped
        data_daily['Scta'] = signalFinal
        data_daily = data_daily.assign(Scta=lambda df: df.Scta.shift(0))[
            ['Scta', 'high', 'close', 'open', 'low', 'volume', 'contract_id', 'price_return']]

        data_daily['DATE_TIME'] = data_daily.index
        # data_daily['TIME'] = data_daily['DATE_TIME'].apply(lambda x: x.strftime('%H:%M:%S'))
        data_daily['trade_date'] = data_daily['DATE_TIME'].apply(lambda x: x.strftime('%Y-%m-%d'))
        data_daily.index = data_daily.DATE_TIME.tolist()
        return data_daily

    def run_test(self, startDate, endDate, **kwargs):
        print(DataFactory.get_mongo_client().database_names())
        self._get_market_info()
        # self._get_history(startDate=startDate, endDate=endDate)
        self._get_history_daily(startDate=startDate, endDate=endDate)
        capital = kwargs['capital']

        signal_dataframe = []
        for symbol in self._symbols:
            market_data_dict = {}
            market_data_dict[symbol] = self._format_data(symbol)
            signal_dataframe.append(self.gen_signal(market_data_dict=market_data_dict, symbol=symbol, capital=capital))
        signal_dataframe = pd.concat(signal_dataframe)
        # if self.result_fold is not None:
        #     signal_dataframe.to_csv(self.result_fold + '\\' + 'signal_dataframe.csv')
        return signal_dataframe