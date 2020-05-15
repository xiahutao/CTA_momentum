# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 16:44
# @Author  : zhangfang
# -*- coding: utf-8 -*-
import sys
import os
import datetime
import pandas as pd

CurrentPath = os.path.dirname(__file__)
# print(CurrentPath)
sys.path.append(CurrentPath.replace('backtest_config_cta_momentum', ''))
# print(sys.path)
import warnings
import time

warnings.filterwarnings("ignore")
import traceback

import data_engine.setting as setting
from data_engine.data_factory import DataFactory
from common.file_saver import file_saver
from common.decorator import runing_time
from data_engine.market_tradingdate import Market_tradingdate
from data_engine.instrument.future import Future
import data_engine.global_variable as global_variable

t1 = time.clock()
from common.logger import logger
from common.file_saver import file_saver
from config.config import Config_back_test
import data_engine.setting as setting
from data_engine.setting import ASSETTYPE_FUTURE
from data_engine.data_factory import DataFactory
from analysis.execution_settle_analysis import execution_settle_analysis
from backtest_config_cta_momentum.ctamomentum import CtaMomentumStrategy_ex

import time
import datetime
import data_engine.setting as setting
from common.os_func import check_fold
from common.file_saver import file_saver
from common.logger import logger
from data_engine.data_factory import DataFactory
from data_engine.setting import ASSETTYPE_FUTURE
from config.config import Config_back_test
from backtest_config_cta_momentum.ctamomentum import CtaMomentumStrategy_ex,get_trade_days,get_day_and_night_symbols
# @runing_time
def run_backtest(symbol, config_file_yaml, saving_file=False):

    try:
        DataFactory.config(MONGDB_PW='jz501241', MONGDB_USER='dbmanager_future',
                           DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE)
        each = symbol
        fut = Future(each)
        result_folder='D://strategy_signal//bactest_result//resCTAMomentum//' + fut.sector
        # 策略对象

        config = Config_back_test(config_path=os.path.split(__file__)[0]).load(config_file=config_file_yaml)
        config.strategy_config['end_date'] = datetime.date(2199, 1, 1)
        config.set_result_folder(result_folder=result_folder + '//resRepo_momentum_all_%sM_exec_%s_%s' % (
        config.get_strategy_config('period'), each, config.get_execution_config('exec_lag')))
        config.strategy_id = 'Momentum-Daily'
        # 回测
        stragegy_obj = CtaMomentumStrategy_ex(config=config, curr_date=curr_date, asset_type=ASSETTYPE_FUTURE,
                                               symbol=each)
        signal_dataframe = stragegy_obj.run_test()
        # signal_dataframe = signal_dataframe.loc[ '2018-01-01':, ]

        analysis_obj = execution_settle_analysis(signal_dataframe=signal_dataframe, config=config)
        if analysis_obj is not None:
            # 保存目标持仓
            target_position_dataframe = analysis_obj.settlement_obj._positions_dataframe
            # if not target_position_dataframe.empty:
                # target_position_dataframe.to_csv('e:/target_position_dataframe.csv')
                # analysis_obj.save_result_stragety_log(collection_name='target_position',
                #                                       series_tmp=target_position_dataframe.ffill().dropna(
                #                                           subset=['position']))
            # 保存pnl, returns等
            analysis_obj.save_result()
            analysis_obj.plot_all()
            fromdate = global_variable.get_now() + datetime.timedelta(days=-365*20)
            fromdate = None
            stragegy_obj.upload_dailyreturn(daily_return=analysis_obj.settlement_obj.daily_return_by_init_aum,fromdate=fromdate
                                             )
            stragegy_obj.upload_volatility(daily_return=analysis_obj.settlement_obj.daily_return_by_init_aum,fromdate=fromdate
                                                )
    except:
        if saving_file:
            file_saver().join()
        traceback.print_exc()
    if saving_file:
        file_saver().join()
    logger().debug('=================', 'run_backtest', '%.6fs' % (time.clock() - t1))



if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz501241', MONGDB_USER='dbmanager_future',
                       DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE)
    from multiprocessing import Pool, cpu_count
    import datetime
    from common.os_func import check_fold
    from common.logger import logger
    from config.config import Config_back_test
    from portfolio_management.strategy import Strategy
    from strategy.strategy import strategy_helper
    from data_engine.market_tradingdate import Market_tradingdate

    tradingdates = Market_tradingdate('SHF')
    tradingdates.get_hisdata()

    last_trading_date = tradingdates.get_last_trading_date(date=global_variable.get_now(),include_current_date=False)
    if datetime.datetime.now().hour < 15:
        curr_date = last_trading_date.strftime('%Y-%m-%d')
    else:
        curr_date = datetime.datetime.now().strftime('%Y-%m-%d')
    curr_date = '2020-03-30'
    next_trading_date = tradingdates.get_next_trading_date(date=pd.to_datetime(curr_date),include_current_date=False)

    trd_day_df, trd_day_after_curr = get_trade_days(curr_date)

    log = None
    cpu = cpu_count() - 1
    cpu = max(1, cpu)

    for config_file_yaml in ['config_sample_1d.yaml']:
        config = Config_back_test(config_path=os.path.split(__file__)[0]) \
            .load(config_file=config_file_yaml)
        run_symbols = []
        products = config.get_strategy_config('product_group')

        # for product in products:
        #     for product in product_list:
        #         run_symbols.append(product)
        try:
            s = Strategy(strategy_name='Momentum-Daily')
            agg = s.list_aggtoken()
            hasnewagg=False
            for product in [x + '_VOL' for x in products]:
                if product not in agg:
                    s.register_aggtoken(aggtoken=product,target_vol=0.15)
                    hasnewagg=True
            if hasnewagg:
                s.update_weight()

            for product in [x + '_VOL' for x in products]:
                run_backtest(product, config_file_yaml, saving_file=False)
            DataFactory().clear_data()
        except:
            DataFactory().clear_data()
            traceback.print_exc()

        file_saver().join()
