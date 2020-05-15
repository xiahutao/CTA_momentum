#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/2/25 15:02
# @Author  : jwliu
# @Site    : 
# @Software: PyCharm

# -*- coding: utf-8 -*-
import sys
import os

CurrentPath = os.path.dirname(__file__)
# print(CurrentPath)
sys.path.append(CurrentPath.replace('backtest', ''))
# print(sys.path)
import warnings
import time
import pandas

warnings.filterwarnings("ignore")
import traceback

import data_engine.setting as setting
from data_engine.data_factory import DataFactory
from common.file_saver import file_saver
from common.decorator import runing_time
import copy
import datetime
import data_engine.global_variable as global_variable
from common.logger import logger
from common.file_saver import file_saver
from config.config import Config_back_test
import data_engine.setting as setting
from data_engine.setting import ASSETTYPE_FUTURE
from data_engine.data_factory import DataFactory
from analysis.execution_settle_analysis import execution_settle_analysis
from ctamomentum1 import CtaCtaStrategy_ex


def get_best_arg_df(best_args_path, result_filename):
    '''
    加载优化参数的文件， 生成参数表的dataframe,按时间排序
    :param best_args_path:
    :param result_filename:
    :return:
    '''
    series_list = []
    for x in os.walk(best_args_path):
        for f in x[2]:
            if result_filename in f:
                series = pandas.read_csv(os.path.join(x[0], f), header=None, names=['key', f]).set_index('key')[f]
                if ':00:00' in series['look_back_end']:
                    series['look_back_end'] = pandas.to_datetime(series['look_back_end']).date()
                    # series['look_back_end'] = datetime.datetime.strptime(series['look_back_end'],
                    #                                                      '%Y-%m-%d %H:%M:%S').date()
                else:
                    series['look_back_end'] = datetime.datetime.strptime(series['look_back_end'], '%Y-%m-%d').date()
                series_list.append(series)
    best_arg_df = pandas.concat(series_list, axis=1).T
    best_arg_df['date_index'] = best_arg_df['look_back_end']
    best_arg_df = best_arg_df.set_index('date_index')
    best_arg_df.index = pandas.to_datetime(best_arg_df.index)
    return best_arg_df.sort_index()


def prepare_strategy_obj(run_pairs
                         , path
                         , start_date=None, end_date=None
                         , config_file_yaml='config_sample_1d.yaml'
                         , saving_file=False
                         ):
    '''
    准备pair策略对象， 调用load_data加载全部历史数据
    :param run_pairs:
    :param path:
    :param start_date:
    :param end_date:
    :param config_file_yaml:
    :param saving_file:
    :return:
    '''
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
    # DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE,
    #                    logging_level=global_variable.logging.INFO)
    curr_date = datetime.date.today().strftime('%Y-%m-%d')

    try:
        config = Config_back_test().load(config_file=config_file_yaml)
        config.strategy_config['start_date'] = start_date
        config.strategy_config['end_date'] = end_date

        # config.strategy_config['volLookback'] = volLookback
        # config.strategy_config['speeds'] = speeds
        # config.strategy_config['scaler'] = scaler
        # config.strategy_id = None
        config.set_result_folder(os.path.join(path, '_'.join(run_pairs)))
        print(run_pairs, config.result_folder)
        strategy_obj = CtaCtaStrategy_ex(config=config, curr_date=curr_date, symbol_list=run_pairs,
                                                asset_type=global_variable.ASSETTYPE_FUTURE)
        strategy_obj.load_data(startDate=None, endDate=None)
        return strategy_obj

        # signal_dataframe = strategy_obj.run_test(startDate=config.get_strategy_config('start_date'), endDate=config.get_strategy_config('end_date'))
        # return signal_dataframe
    except:
        if saving_file:
            file_saver().join()
        traceback.print_exc()
    file_saver().join()
    logger().debug('=================', 'run_backtest', '%.6fs' % (time.clock()))


if __name__ == '__main__':
    # DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE,
                       logging_level=global_variable.logging.INFO)

    from multiprocessing import Pool, cpu_count
    import datetime
    from common.os_func import check_fold
    from common.logger import logger
    from config.config import Config_back_test
    import data_engine.global_variable as global_variable

    current_date = datetime.datetime.now()

    # 回测配置文件
    config_file_yaml = 'config_sample_1d.yaml'
    config = Config_back_test(config_path=os.path.split(__file__)[0]) \
        .load(config_file=config_file_yaml)
    run_pairs = []
    products = config.get_strategy_config('product_group')

    run_pairs_0 = [tuple([y + '_VOL' for y in x]) for x in products]
    run_pairs_0 = ['A', 'AG', 'AL', 'AP', 'AU', 'B', 'BU', 'C', 'CF', 'CS', 'CU', 'FG',
                   'HC', 'I', 'IC', 'IF', 'IH', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P',
                   'PB', 'PP', 'RB', 'RM', 'RU', 'SC', 'SF', 'SM', 'SN', 'SR',]  # 'C'\'I'\'T'
    run_pairs_0 = ['TA', 'TF', 'V', 'Y', 'ZC', 'ZN']  # 所有品种
    run_pairs_0 = ['C', 'I', 'T']

    # 优化参数文件保存路径
    best_args_path = r'E:/Strategy/MOMENTUM/best_args'

    for each in run_pairs_0:
        run_pairs = [each]

        # 回测结果保存路径
        result_path = os.path.join(r'e:/Strategy/MOMENTUM/MOMENTUM_args_opt/opt', '_'.join(run_pairs))
        config.set_result_folder(result_path)

        # 优化参数记录
        result_filename = 'best_arg_series' + '_'.join(run_pairs)
        best_arg_df = get_best_arg_df(best_args_path=best_args_path, result_filename=result_filename)
        if best_arg_df.empty:
            continue
        file_saver().save_file(best_arg_df, os.path.join(result_path, 'best_arg_df.csv'))

        # 生成策略对象，一次加载背景行情数据
        strategy_obj = prepare_strategy_obj(run_pairs
                                            , path=result_path, saving_file=False)

        signal_dataframe_list = []
        for idx, row in best_arg_df.iterrows():
            # 优化日 look_back_end， 交易look_back_end到下月1日
            fromdate = row['look_back_end']
            todate = fromdate + datetime.timedelta(days=31)
            todate = datetime.date(todate.year, todate.month, 1)

            # 优化参数
            s_period = [int(row['s_period'] * 3)]
            l_period = [int(row['l_period'] * 3) + int(row['s_period'] * 3)]
            vlb = int(row['volLookback']) * 20

            # vlb = 60
            # sl= 3
            # spd = 6
            # 获得fromdate到todate的信号
            signal_dataframe = strategy_obj.run_test(startDate=fromdate
                                                     , endDate=todate
                                                     , config=config
                                                     , s_period=s_period
                                                     , l_period=l_period
                                                     , volLookback=vlb
                                                     )
            # signal_dataframe = signal_dataframe[(signal_dataframe.index >= fromdate) & (signal_dataframe.index <= todate)]

            signal_dataframe.index.name = 'date_index'
            signal_dataframe_list.append(signal_dataframe.reset_index())

        # 拼接signal_dataframe， 进行撮合分析保存回测结果
        signal_dataframe = pandas.concat(signal_dataframe_list)
        signal_dataframe = signal_dataframe.drop_duplicates(subset=['date_index', 'symbol'], keep='last').set_index(
            'date_index')
        file_saver().save_file(signal_dataframe, os.path.join(result_path, 'signal_dataframe.csv'))

        analysis_obj = execution_settle_analysis(signal_dataframe=signal_dataframe, config=config)
        if analysis_obj is not None:
            print(analysis_obj._result_folder)
            # 保存目标持仓
            target_position_dataframe = analysis_obj.settlement_obj._positions_dataframe
            analysis_obj.plot_all()
            # 保存pnl, returns等
            analysis_obj.save_result(only_returns=True)
    file_saver().join()
