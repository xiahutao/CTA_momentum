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
from backtest_config_cta_momentum.ctamomentum import CtaMomentumStrategy_ex
# @runing_time
def run_backtest(symbol, config_file_yaml, saving_file=False):

    try:
        DataFactory.config(MONGDB_PW='jz501241', MONGDB_IP='192.168.2.201', MONGDB_USER='dbmanager_future',
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
            # analysis_obj.plot_all()
            stragegy_obj.load_strategy_aggtoken_his(aggtoken=stragegy_obj.aggToken)
            stragegy_obj.upload_dailyreturn(daily_return=analysis_obj.settlement_obj.daily_return_by_init_aum
                                             )
            stragegy_obj.upload_volatility(daily_return=analysis_obj.settlement_obj.daily_return_by_init_aum
                                                )
    except:
        if saving_file:
            file_saver().join()
        traceback.print_exc()
    if saving_file:
        file_saver().join()
    logger().debug('=================', 'run_backtest', '%.6fs' % (time.clock() - t1))


def get_trade_days(curr_date):
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)

    trd_day_df = Market_tradingdate('SHFE').HisData_DF[['Tradedays_str', 'isTradingday']]
    trd_day_df = trd_day_df[trd_day_df['isTradingday'] == True]
    trd_day_after_curr = trd_day_df[trd_day_df['Tradedays_str'] > curr_date].Tradedays_str.iloc[0]
    return trd_day_df, trd_day_after_curr


def get_day_and_night_symbols(symbols_lst):
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
    night_symbols = []
    day_symbols = []
    for symbol in symbols_lst:
        ts = Future(symbol).get_trading_sessions()
        tradeNightStartTime = ts.Session4_Start.zfill(8)
        tradeNightEndTime = ts.Session4_End.zfill(8)
        if tradeNightEndTime < '04:00:00' or tradeNightEndTime >= '23:30:00':
            tradeNightEndTime = '23:30:00'
        if tradeNightEndTime != tradeNightStartTime:
            night_symbols.append(symbol)
        else:
            day_symbols.append(symbol)
    return night_symbols, day_symbols

# @runing_time
def gen_config(product, config_file_yaml, curr_date, log_file=None, config_file=r'e://Strategy//MOMENTUM//configs',
               saving_file=False,exist_and_return=False,append=False):
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
    run_symbols=[product]
    if log_file is None:
        log_file = os.path.join(config_file, 'log', 'momentum_' + datetime.datetime.now().strftime(
            '%Y%m%d') + '.log')
    log = logger.get_logger(log_file)
    t1 = time.clock()
    try:
        config = Config_back_test(config_path=os.path.split(__file__)[0]).load(config_file=config_file_yaml)
        trd_day_df, trd_day_after_curr = get_trade_days(curr_date)
        if curr_date not in trd_day_df.Tradedays_str:
            print('{} 今天是非交易日'.format(curr_date))
            return
        config.strategy_config['end_date'] = curr_date
        night_symbols, day_symbols = get_day_and_night_symbols(run_symbols)

        for each in run_symbols:
            symbol = each
            if each in night_symbols:
                method = 'night'
            else:
                method = 'day'
            configFile = r'{0}\{1}_{2}_{3}_{4}.csv'.format(
                config_file, config.strategy_config['strategy_type'],
                curr_date, each, method)

            if os.path.exists(configFile) and exist_and_return:
                continue
            DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
            config_ret = CtaMomentumStrategy_ex(config=config, curr_date=curr_date,
                                               asset_type=ASSETTYPE_FUTURE,
                                                symbol=symbol, method=method) \
                .run_cofig()
            check_fold(config_file)
            # 保存配置信息
            config_ret.dump_csv(config_file=configFile,append=append)
    except:
        if saving_file:
            file_saver().join()
        traceback.print_exc()
    if saving_file:
        file_saver().join()
    DataFactory().clear_data()
    log.debug('=================', 'gen_config', '%.6fs' % (time.clock() - t1))


if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz501241', MONGDB_IP='192.168.2.201', MONGDB_USER='dbmanager_future',
                       DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE)
    from multiprocessing import Pool, cpu_count
    import datetime
    from common.os_func import check_fold
    from common.logger import logger
    from config.config import Config_back_test
    from portfolio_management.strategy import Strategy

    curr_date = datetime.date.today().strftime('%Y-%m-%d')
    curr_date = '2020-03-27'
    trd_day_df, trd_day_after_curr = get_trade_days(curr_date)

    log = None
    cpu = cpu_count() - 1
    cpu = max(1, cpu)
    for config_file_yaml in ['config_sample_1d.yaml']:
        config = Config_back_test(config_path=os.path.split(__file__)[0]) \
            .load(config_file=config_file_yaml)

        if not config.strategy_config['runing_model'].upper() == 'SIMNOW':
            continue
        stragety_name = config.strategy_config['strategy_type']
        run_symbols = []
        products = config.get_strategy_config('product_group')

        for product_list in products:
            for product in product_list:
                run_symbols.append(product)

        config_file = config.get_strategy_config('config_file')  # r'D:\strategy_signal\signal_config'
        config_file_output = config.get_strategy_config('config_file_output')  # r'D:\strategy_signal\signal_config'
        check_fold(os.path.join(config_file, 'temp'))
        check_fold(os.path.join(config_file, 'log'))
        log_file = os.path.join(config_file, 'log', stragety_name + '_' + datetime.datetime.now().strftime('%Y%m%d') + '.log')
        log = logger().with_file_log(log_file).with_scream_log().setLevel()
        log.debug('Start:  single ' + stragety_name + '_' + datetime.datetime.now().strftime('%Y%m%d') + '.csv')

        pool = Pool(cpu, maxtasksperchild=3)
        try:
            log.debug('apply_async: symbols' + '_'.join(products))
            for product in [x + '_VOL' for x in products]:
                pool.apply_async(gen_config,args=(product, config_file_yaml, curr_date, log_file, os.path.join(config_file, 'temp'), False, False))
                # gen_config(product, config_file_yaml, curr_date, log_file, os.path.join(config_file, 'temp'), False, False)
            DataFactory().clear_data()
        except:
            DataFactory().clear_data()
            traceback.print_exc()
        pool.close()
        pool.join()

        try:
            for product in [x + '_VOL' for x in products]:
                gen_config(product, config_file_yaml, curr_date, log_file, os.path.join(config_file, 'temp'), saving_file=False,exist_and_return=True)
            DataFactory().clear_data()
        except:
            DataFactory().clear_data()
            traceback.print_exc()
        file_saver().join()
        log.debug('Done:  ' + stragety_name + ' symbols_' + config.get_data_config('freq') + '_' +
                  datetime.datetime.now().strftime('%Y%m%d') + '.csv')
        for x in os.walk(os.path.join(config_file, 'temp')):
            lines_day = []
            lines_night = []
            for f in x[2]:
                if curr_date in f and 'night' in f:
                    with open(os.path.join(x[0], f), encoding='utf-8') as file:
                        line = file.readline()
                        lines_night.append(line)
                        file.close()
                elif curr_date in f and 'day' in f:
                    with open(os.path.join(x[0], f), encoding='utf-8') as file:
                        line = file.readline()
                        lines_day.append(line)
                        file.close()
            configFile_day = r'{0}\{1}_{2}_day.csv'.format(config_file_output, 'cta_daily_requests',
                                                           pd.to_datetime(trd_day_after_curr).strftime('%Y%m%d'))
            configFile_night = r'{0}\{1}_{2}_night.csv'.format(config_file_output, 'cta_daily_requests',
                                                               pd.to_datetime(curr_date).strftime('%Y%m%d'))
            fp_day = open(configFile_day, mode='w')
            fp_day.writelines(lines_day)
            fp_day.close()

            fp_night = open(configFile_night, mode='w')
            fp_night.writelines(lines_night)
            fp_night.close()

        log.debug('Done:  ' + stragety_name + config.get_data_config('freq') + '_' +
                  datetime.datetime.now().strftime('%Y%m%d') + '.csv')

        log.insert_record_to_updateLog(machine_IP='119.3.39.142', task_name=stragety_name, job=__file__)

    for config_file_yaml in ['config_sample_1d.yaml']:
        config = Config_back_test(config_path=os.path.split(__file__)[0]) \
            .load(config_file=config_file_yaml)
        if not config.strategy_config['runing_model'].lower() == 'backtest':
            continue
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
