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
import numpy
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
import yaml
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
def gen_config(product, config_file_yaml, curr_date, log_file=None, config_file=r'e://Strategy//MOMENTUM//configs',
               saving_file=False,exist_and_return=False):
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
    temp_folder = os.path.join(config_file, 'temp')
    check_fold(temp_folder)
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
                configFile = r'{0}\{1}_{2}_{3}_{4}.csv'.format(
                    temp_folder, config.strategy_config['strategy_type'],
                    curr_date, each, method)
            else:
                method = 'day'
                configFile = r'{0}\{1}_{2}_{3}_{4}.csv'.format(
                    temp_folder, config.strategy_config['strategy_type'],
                    trd_day_after_curr, each, method)

            strategy_obj = CtaMomentumStrategy_ex(config=config, curr_date=curr_date,
                                               asset_type=ASSETTYPE_FUTURE,
                                                symbol=symbol, method=method)
            config_ret,config_if_main_contract_switch = strategy_obj.run_cofig()

            #main_contract_switch
            if len(config_if_main_contract_switch)>0:
                for each_config in config_if_main_contract_switch:
                    if strategy_obj._product.has_night_trading(bydate=curr_date):
                        configFileName = '_'.join([config.strategy_config['master_config_prefix'],
                                                   curr_date, 'night',strategy_obj._strategy_name,strategy_obj.aggToken,'mcs']) + '.csv'
                    else:
                        configFileName = '_'.join([config.strategy_config['master_config_prefix'],
                                                   trd_day_after_curr, 'day',strategy_obj._strategy_name,strategy_obj.aggToken,'mcs']) + '.csv'
                    each_config.dump_csv(config_file = os.path.join(temp_folder,configFileName),append=False)


            # 保存配置信息
            if not config_ret is None:
                config_ret.dump_csv(config_file=configFile,append=False)
    except:
        if saving_file:
            file_saver().join()
        traceback.print_exc()
    if saving_file:
        file_saver().join()
    DataFactory().clear_data()
    log.debug('=================', 'gen_config', '%.6fs' % (time.clock() - t1))


if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
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

        stragety_name = config.strategy_config['strategy_type']
        run_symbols = []

        config_file = config.get_strategy_config('config_file')  # r'D:\strategy_signal\signal_config'
        config_file_output = config.get_strategy_config('config_file_output')  # r'D:\strategy_signal\signal_config'
        check_fold(os.path.join(config_file, 'temp'))
        check_fold(os.path.join(config_file, 'log'))
        log_file = os.path.join(config_file, 'log', stragety_name + '_' + datetime.datetime.now().strftime('%Y%m%d') + '.log')
        log = logger().with_file_log(log_file).with_scream_log().setLevel()
        log.debug('Start:  single ' + stragety_name + '_' + datetime.datetime.now().strftime('%Y%m%d') + '.csv')

        products = config.get_strategy_config('product_group')
        # pool = Pool(cpu, maxtasksperchild=3)

        try:
            log.debug('apply_async: symbols' + '_'.join(products))
            for product in [x + '_VOL' for x in products]:
                # pool.apply_async(gen_config,
                #                  args=(product, config_file_yaml, curr_date, log_file, config_file, False, False))
                gen_config(product, config_file_yaml, curr_date, log_file, config_file, False, False)
            DataFactory().clear_data()
        except:
            DataFactory().clear_data()
            traceback.print_exc()
        # pool.close()
        # pool.join()

        file_saver().join()
        log.debug('Done:  ' + stragety_name + ' symbols_' + config.get_data_config('freq') + '_' +
                  datetime.datetime.now().strftime('%Y%m%d') + '.csv')
        # for x in os.walk(os.path.join(config_file, 'temp')):
        #     lines_day = []
        #     lines_night = []
        #     for f in x[2]:
        #         if curr_date in f and 'night' in f:
        #             with open(os.path.join(x[0], f), encoding='utf-8') as file:
        #                 line = file.readline()
        #                 lines_night.append(line)
        #                 file.close()
        #         elif curr_date in f and 'day' in f:
        #             with open(os.path.join(x[0], f), encoding='utf-8') as file:
        #                 line = file.readline()
        #                 lines_day.append(line)
        #                 file.close()
        #     configFile_day = r'{0}\{1}_{2}_day.csv'.format(config_file_output, 'cta_daily_requests',
        #                                                    pd.to_datetime(trd_day_after_curr).strftime('%Y%m%d'))
        #     configFile_night = r'{0}\{1}_{2}_night.csv'.format(config_file_output, 'cta_daily_requests',
        #                                                        pd.to_datetime(curr_date).strftime('%Y%m%d'))
        #     fp_day = open(configFile_day, mode='w')
        #     fp_day.writelines(lines_day)
        #     fp_day.close()
        #
        #     fp_night = open(configFile_night, mode='w')
        #     fp_night.writelines(lines_night)
        #     fp_night.close()

        #合并temp目录的文件统一输出
        config_day_lines, config_night_lines, stoploss_lines = strategy_helper.merge_config_file(
                                                                        config_file_path=config.get_strategy_config('config_file'),
                                                                        config_file_output_path=config.get_strategy_config('config_file_output'),
                                                                        stoploss_file_output_path=config.get_strategy_config('stoploss_file_output'),
                                                                        current_business_date=curr_date, next_business_date = next_trading_date
                                                                    )

        config = Config_back_test(config_path=os.path.split(__file__)[0]).load(config_file=config_file_yaml)
        tmp_obj = CtaMomentumStrategy_ex(curr_date=datetime.datetime.now(), config=config,
                                         asset_type=global_variable.ASSETTYPE_FUTURE, symbol='RB_VOL')
        # mail_to_list = ['519518384@qq.com','49680664@qq.com']
        #
        # tmp_obj.check_and_mail(mail_to_list=mail_to_list
        #                        ,config_day_lines=config_day_lines, config_night_lines=config_night_lines, stoploss_lines=stoploss_lines)


        log.debug('Done:  ' + stragety_name + config.get_data_config('freq') + '_' +
                  datetime.datetime.now().strftime('%Y%m%d') + '.csv')

        log.insert_record_to_updateLog(machine_IP='119.3.39.142', task_name=stragety_name, job=__file__)
