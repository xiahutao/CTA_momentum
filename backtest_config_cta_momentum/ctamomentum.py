# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 16:45
# @Author  : zhangfang


import pandas as pd
import numpy  as np
import datetime
from common.file_saver import file_saver
from common.decorator import runing_time
from strategy.strategy import Strategy
from data_engine.data_factory import DataFactory
from data_engine.global_variable import ASSETTYPE_FUTURE, DATASOURCE_REMOTE, DATASOURCE_LOCAL
from config.config import Config_back_test, Config_trading

import data_engine.global_variable as global_variable
from data_engine.instrument.future import Future
from data_engine.instrument.product import Product
from data_engine.market_tradingdate import Market_tradingdate

def get_trade_days(curr_date):
    trd_day_df = Market_tradingdate('SHFE').HisData_DF[['Tradedays_str', 'isTradingday']]
    trd_day_df = trd_day_df[trd_day_df['isTradingday'] == True]
    trd_day_after_curr = trd_day_df[trd_day_df['Tradedays_str'] > curr_date].Tradedays_str.iloc[0]
    return trd_day_df, trd_day_after_curr


def get_day_and_night_symbols(symbols_lst):
    night_symbols = []
    day_symbols = []
    for symbol in symbols_lst:
        ts = Future(symbol)#.get_trading_sessions()
        # tradeNightStartTime = ts.Session4_Start.zfill(8)
        # tradeNightEndTime = ts.Session4_End.zfill(8)
        # if tradeNightEndTime < '04:00:00' or tradeNightEndTime >= '23:30:00':
        #     tradeNightEndTime = '23:30:00'
        if ts.has_night_trading():
            night_symbols.append(symbol)
        else:
            day_symbols.append(symbol)
    return night_symbols, day_symbols

class CtaMomentumStrategy(Strategy):
    # _strategy_type = 'intraday_pair'
    def __init__(self, curr_date, asset_type, symbol, freq, s_period, l_period, targetVol, maxLeverage,
                 period, result_fold, method, **kwargs):

        super(CtaMomentumStrategy, self).__init__(period=period,
                                                  s_period=s_period,
                                                  l_period=l_period,
                                                  targetVol=targetVol,
                                                  maxLeverage=maxLeverage,
                                                  method=method,
                                                  strategy_name='Momentum-Daily')
        self.aggToken = symbol
        self._freq = freq
        self._asset_type = asset_type
        self.symbol = symbol
        # self.curr_date = datetime.date.today().strftime('%Y-%m-%d')
        self.curr_date = curr_date
        self.method = method

        self._symbols = set()
        for s1 in [symbol]:
            if s1 not in self._symbols:
                self._symbols.add(s1)

        self._instrument = Future(symbol=symbol)
        self._product = Product(self._instrument.product_id)
        self._market_data = None
        self.result_fold = result_fold

    def _get_history(self, startDate, endDate, **kwargs):
        self._market_data = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq=self._freq,
                                                          symbols=self.symbol, end_date=endDate)

    def _get_history_daily(self, startDate, endDate):
        self._market_data_daily = DataFactory().get_market_data(asset_type=ASSETTYPE_FUTURE, freq='1d',
                                                                symbols=self.symbol, start_date=startDate,
                                                                end_date=endDate)

    def get_trade_days(self):
        trd_day = Market_tradingdate('SHFE').HisData_DF[['Tradedays_str', 'isTradingday']]
        trd_day = trd_day[trd_day['isTradingday'] == True]
        self.trd_day = trd_day
        self.trd_day_after_curr = trd_day[trd_day['Tradedays_str'] > self.curr_date].Tradedays_str.iloc[0]

    def vol_estimator_garch(self, data_df, st=25, lt=252 * 3):
        st_vol = data_df.ewm(span=st, ignore_na=True, min_periods=st, adjust=False).std(bias=True)
        lt_vol = data_df.ewm(span=lt, ignore_na=True, min_periods=st, adjust=False).std(bias=True)
        decay_rate = 0.8
        vol = st_vol * decay_rate + lt_vol * (1 - decay_rate)
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

    def get_position(self, symbol, dailyPrice, **kwargs):
        capital = kwargs['capital']
        targetVol = kwargs['target_vol']
        print(targetVol)
        SK = self._params['s_period']
        LK = self._params['l_period']
        maxLeverage = self._params['maxLeverage']
        fut = Future.get_future(symbol=symbol)
        dailyPrice_return = dailyPrice.price_return.fillna(value=0)
        for k in range(len(SK)):
            s_period = SK[k]
            l_period = LK[k]
            volAdjRet = dailyPrice_return / dailyPrice_return.ewm(span=s_period, min_periods=s_period,
                                                                  adjust=False).std(bias=True)
            px_df = np.cumsum(volAdjRet)
            sig = px_df.ewm(span=s_period, min_periods=s_period).mean() - px_df.ewm(span=l_period,
                                                                                    min_periods=s_period).mean()
            sig_normalized = sig / self.vol_estimator_garch(sig, 25)
            sig_resp = self.get_resp_curve(sig_normalized, 'gaussian')
            sig = sig_normalized * sig_resp * 1.0 / 0.89
            realizedVol = dailyPrice_return.ewm(span=20, ignore_na=True, min_periods=20,
                                                adjust=False).std(bias=True) * (252 ** 0.5)
            if fut.product_id.upper() not in ['T', 'TF']:
                realizedVol = self.cap_vol_by_rolling(realizedVol)
            riskScaler = targetVol / realizedVol
            scaledSig = riskScaler * sig
            dailyPrice['uk' + str(k)] = scaledSig
            dailyPrice['uk' + str(k)].fillna(method='pad', inplace=True)
        scaledSig = 1 / 5 * (
                dailyPrice['uk0'] + dailyPrice['uk1'] + dailyPrice['uk2'] + dailyPrice['uk3'] + dailyPrice['uk4'])
        leverageRaw = abs(scaledSig)
        leverageCapped = leverageRaw.apply(lambda x: min(x, maxLeverage))
        signOfSignal = scaledSig.apply(lambda x: np.sign(x))
        signalFinal = signOfSignal * leverageCapped
        position = signalFinal * capital / self._contract_size_dict[symbol] / dailyPrice['close']
        position_trd = position[-1]
        f = Future(dailyPrice.contract_id.iloc[-1])
        instrument = f.ctp_symbol
        market = f.market
        dailyPrice['date_time'] = dailyPrice.index
        histLastSignalTime = dailyPrice.date_time.iloc[-1]
        return position_trd, instrument, market, histLastSignalTime

    def gen_cofig(self, **kwargs):

        capital = kwargs['capital']
        targetVol = kwargs['target_vol']
        config = Config_trading()
        config_if_main_contract_switch = []

        symbol = self.symbol
        f = self._instrument
        p = Product(product_id=f.product_id).load_futures()
        p.load_hq()
        p.get_hq_panel()
        max_volume_fut = p.max_volume_fut()
        if self.method == 'night':
            print('night symbol {}'.format(symbol))
            ts = self._instrument.get_trading_sessions()
            tradeNightStartTime = ts.Session4_Start.zfill(8)
            dailyPrice = self._market_data_daily[symbol]
            if str(dailyPrice.trade_date.iloc[-1])[:10] < self.curr_date:
                print('========Error: {} 未取得最新行情'.format(symbol))
                return config,config_if_main_contract_switch
            position_trd, instrument, market, histLastSignalTime = self.get_position(symbol, dailyPrice, **kwargs)

            #换月
            if position_trd != 0:
                config_if_main_contract_switch = self.gen_config_if_main_contract_switch(product_obj=self._product,
                                                                                         curr_date_str=self.curr_date,
                                                                                         position_trd=0,
                                                                                         only_close_last_constract=True)


            if max_volume_fut is not None:
                instrument = max_volume_fut.ctp_symbol
            valideStartTime = self.curr_date + ' ' + tradeNightStartTime
            config = self.gen_target_position_config(requestType='Create',
                                                     instrument=instrument,
                                                     market=market,
                                                     aggToken=self.aggToken,
                                                     requestTime=valideStartTime,
                                                     aggregateRequest='true',
                                                     targetPosition=position_trd,
                                                     strategy=self._strategy_name,
                                                     histLastSignalTime=histLastSignalTime,
                                                     initiator='Agg-Proxy',
                                                     capital=capital,
                                                     targetVol=targetVol)
        elif self.method == 'day':
            print('day symbol {}'.format(symbol))
            ts = self._instrument.get_trading_sessions()
            tradeDayStartTime = ts.Session1_Start.zfill(8)
            dailyPrice = self._market_data_daily[symbol]
            if str(dailyPrice.trade_date.iloc[-1])[:10] < self.curr_date:
                print('========Error: {} 未取得最新行情'.format(symbol))
                return config,config_if_main_contract_switch
            position_trd, instrument, market, histLastSignalTime = self.get_position(symbol, dailyPrice, **kwargs)


            valideStartTime = self.trd_day_after_curr + ' ' + tradeDayStartTime

            #换月
            if position_trd != 0:
                config_if_main_contract_switch = self.gen_config_if_main_contract_switch(product_obj=self._product,
                                                                                         curr_date_str=self.trd_day_after_curr,
                                                                                         position_trd=0,
                                                                                         only_close_last_constract=True)

            if max_volume_fut is not None:
                instrument = max_volume_fut.ctp_symbol
            config = self.gen_target_position_config(requestType='Create',
                                                     instrument=instrument,
                                                     market=market,
                                                     aggToken=self.aggToken,
                                                     requestTime=valideStartTime,
                                                     aggregateRequest='true',
                                                     targetPosition=position_trd,
                                                     strategy=self._strategy_name,
                                                     histLastSignalTime=histLastSignalTime,
                                                     initiator='Agg-Proxy',
                                                     capital=capital,
                                                     targetVol=targetVol
                                                     )
        return config,config_if_main_contract_switch

    @runing_time
    def gen_signal(self, **kwargs):
        capital = kwargs['capital']
        market_data_dict = kwargs['market_data_dict']
        symbol = kwargs['symbol']
        pos_df_list = []
        data = market_data_dict[symbol].assign(close_1=lambda df: df.close.shift(1))
        _signal_lst = []
        _signal = 0
        capital_intial = capital / self._instrument.contract_size # self._contract_size_dict[symbol]

        data[symbol] = data['Scta'] * capital_intial / data['close']
        signal = data[symbol]
        target_pos_dict = {}
        target_pos_tmp = signal
        target_pos_tmp.fillna(method='pad', inplace=True)
        target_pos_tmp.name = symbol
        target_pos_dict[symbol] = target_pos_tmp
        if self.result_fold is not None:
            file_saver().save_file(target_pos_tmp, self.result_fold + '\\' + symbol + '_target_pos.csv')

        contract_id_series = data['contract_id']
        contract_id_series.name = symbol + '_contract_id'
        target_pos_df = pd.concat([target_pos_tmp, contract_id_series], axis=1)
        target_pos_dict[symbol] = target_pos_df[symbol]

        pos_serires = target_pos_dict[symbol].copy()
        pos_serires.name = 'position'

        pos_df = pd.DataFrame(pos_serires, index=pos_serires.index)
        pos_df = pos_df.join(contract_id_series)
        pos_df['symbol'] = symbol
        pos_df['asset_type'] = self._asset_type
        pos_df['contract_size'] = self._instrument.contract_size #self._contract_size_dict[symbol]
        pos_df['tick_size'] = self._instrument.tick_size #self._tick_size_dict[symbol]
        pos_df['margin_ratio'] = 0.1
        pos_df['freq'] = self._freq
        pos_df['remark'] = '.'.join([self._strategy_name, self._strategy_type, symbol])

        pos_df_list.append(pos_df)
        signal_dataframe = None
        if len(pos_df_list) > 0:
            signal_dataframe = pd.concat(pos_df_list)
        return signal_dataframe

    @runing_time
    def _format_data(self, symbol, **kwargs):
        targetVol = self.aggtoken_target_vol(aggtoken=self.aggToken) # kwargs['targetVol']
        SK = kwargs['s_period']
        LK = kwargs['l_period']
        maxLeverage = kwargs['maxLeverage']
        data_daily = self._market_data_daily[symbol]
        print(data_daily)
        data_daily = data_daily[data_daily['volume'] > 0][
            ['high', 'close', 'open', 'low', 'volume', 'price_return', 'contract_id']]
        data_daily.price_return = data_daily.price_return.fillna(value=0)
        for k in range(len(SK)):
            volAdjRet = data_daily['price_return'] / data_daily['price_return'].ewm(span=SK[k], min_periods=SK[k],
                                                                                    adjust=False).std(bias=True)
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
        scaledSig = 1 / 5 * (
                    data_daily['uk0'] + data_daily['uk1'] + data_daily['uk2'] + data_daily['uk3'] + data_daily['uk4'])
        leverageRaw = abs(scaledSig)
        leverageCapped = leverageRaw.apply(lambda x: min(x, maxLeverage))
        signOfSignal = scaledSig.apply(lambda x: np.sign(x))
        signalFinal = signOfSignal * leverageCapped
        data_daily['Scta'] = signalFinal
        data_daily['DATE_TIME'] = data_daily.index
        data_daily['trade_date'] = data_daily['DATE_TIME'].apply(lambda x: x.strftime('%Y-%m-%d'))
        data_daily.index = data_daily.DATE_TIME.tolist()
        return data_daily

    @runing_time
    def run_test(self, startDate, endDate, **kwargs):
        self._get_market_info()
        self._get_history_daily(startDate=startDate, endDate=endDate)
        capital = kwargs['capital']
        signal_dataframe = []
        market_data_dict = {}
        for symbol in self._symbols:
            market_data_dict[symbol] = self._format_data(symbol, **kwargs)
            signal_dataframe.append(self.gen_signal(market_data_dict=market_data_dict, symbol=symbol, capital=capital))
        signal_dataframe = pd.concat(signal_dataframe)
        if self.result_fold is not None:
            signal_dataframe.to_csv(self.result_fold + '\\' + 'signal_dataframe.csv')
        return signal_dataframe

    @runing_time
    def run_cofig(self, startDate, endDate, **kwargs):
        self._get_market_info()
        self._get_history_daily(startDate=startDate, endDate=endDate)
        self.get_trade_days()

        # capital = kwargs['capital']
        capital = self.aggtoken_capital(aggtoken=self.aggToken,date=global_variable.get_now())
        # target_vol = kwargs['targetVol']
        target_vol = self.aggtoken_target_vol(aggtoken=self.aggToken)

        market_data_dict = {}
        for symbol in self._symbols:
            data1 = self._market_data_daily[symbol]
            market_data_dict[symbol] = data1
        return self.gen_cofig(market_data_dict=market_data_dict, capital=capital, target_vol=target_vol)


class CtaMomentumStrategy_ex(CtaMomentumStrategy):
    def __init__(self, curr_date, config,
                 asset_type, symbol, method='night'):
        assert isinstance(config, Config_back_test)
        CtaMomentumStrategy.__init__(self, curr_date=curr_date, asset_type=asset_type, symbol=symbol,
                                     freq=config.get_data_config('freq'),
                                     method=method,
                                     result_fold=config.get_result_config('result_folder')
                                     , **config.strategy_config)

        self._config = config

    @runing_time
    def run_cofig(self, **kwargs):
        return CtaMomentumStrategy.run_cofig(self, startDate=self._config.get_strategy_config('start_date'),
                                             endDate=self._config.get_strategy_config('end_date'), method='night',
                                             **self._config.strategy_config, **kwargs)

    @runing_time
    def run_test(self, **kwargs):
        return CtaMomentumStrategy.run_test(self, startDate=self._config.get_strategy_config('start_date'),
                                            endDate=self._config.get_strategy_config('end_date'),
                                            **self._config.strategy_config, **kwargs)
