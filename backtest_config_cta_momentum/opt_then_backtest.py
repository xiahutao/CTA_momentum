#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/3/2 13:47
# @Author  : jwliu
# @Site    : 
# @Software: PyCharm
import sys
import os

CurrentPath = os.path.dirname(__file__)
# print(CurrentPath)
sys.path.append(CurrentPath.replace('backtest', ''))
import datetime

import pandas
import pytz
from common.os_func import check_fold
from common.cache import cache
from multiprocessing import Pool, cpu_count
from analysis.report.script.report_strategy_summary import report as report_strategy_summary
from analysis.report.script.report_strategy_sharpelist import report as report_strategy_sharpelist
from common.cache import cache
from opt.base_opt import BaseOpt
from config.config import Config_back_test
from data_engine.data_factory import DataFactory

import data_engine.global_variable as global_variable
from ctamomentum1 import CtaCtaStrategy_ex
from execution.execution import Execution_ex
from settlement.settlement import Settlement_ex
from analysis.analysis import Analysis_func
from analysis.execution_settle_analysis import execution_settle_analysis
import copy

from opt_arg.opt_momentum_maxsharpe import get_opt_date_df, OptPair_MaxSharpe


def run(each_pair0, config_file_yaml0, best_arg_path):
    DataFactory.config(MONGDB_USER='zhangfang', MONGDB_PW='jz148923', DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE
                       , logging_level=global_variable.logging.INFO
                       , temp_path=r'E:\TEMP_Python\MOMENTUM'
                       )

    opt_result_path = os.path.join(best_arg_path, 'opt_result')
    check_fold(opt_result_path)

    config = Config_back_test().load(config_file=config_file_yaml0)
    # opt_date_list 每个月1日作为优化参数点，产生优化日的列表
    run_pairs = [each_pair0]
    curr_date = datetime.date.today().strftime('%Y-%m-%d')

    namespace = '_'.join(['CtaMomentumStrategy'] + list(run_pairs))
    if not cache.load(namespace=namespace):
        strategy_obj = CtaCtaStrategy_ex(config=config, curr_date=curr_date, symbol_list=run_pairs,
                                      asset_type=global_variable.ASSETTYPE_FUTURE)
        strategy_obj.load_data(startDate=None, endDate=None)
        cache.set_object('strategy_obj', strategy_obj, namespace=namespace)
    else:
        strategy_obj = cache.get_object('strategy_obj', namespace=namespace)

    start_date = strategy_obj.start_date
    opt_date_df = get_opt_date_df(start_date=start_date, dropna=True)

    ret_list = []
    sub_dailyreturns_list = []
    volLookback = 1
    s_period = 4
    l_period = 8
    result_filename = 'benchmark_arg_series' + '_'.join(run_pairs) + '.csv'

    testopt1 = OptPair_MaxSharpe(look_back_start=None, look_back_end=None, validate_end=None, curr_date=curr_date,
                                 run_pairs=run_pairs,
                                 config_file_yaml=config_file_yaml0,
                                 n_jobs=1, n_trials=100)
    # testopt1.get_strategy_obj()
    daily_return_by_init_aum = testopt1.get_dailyreturns(volLookback, s_period, l_period, re_calc=False)
    sub_dailyreturns = Analysis_func.cut_returns(daily_returns=daily_return_by_init_aum
                                                 , look_back_start=None
                                                 , look_back_end=opt_date_df.iloc[0]['opt_date'])
    sub_dailyreturns_list.append(sub_dailyreturns)
    for _, row in opt_date_df.iterrows():
        look_back_end = row['last_opt_date']
        validate_end = row['opt_date']
        next_opt_date = row['next_opt_date']
        look_back_start = look_back_end + datetime.timedelta(days=-365)

        # 优化参数保存文件名
        result_filename = 'best_arg_series' + '_'.join(run_pairs) + '_' + look_back_end.strftime('%Y%m%d') + '.csv'
        # if os.path.exists(os.path.join(best_arg_path, result_filename)):
        #     continue
        # n_jobs = -1 ， 使用全部cpu进行优化迭代
        # n_trials，
        testopt1 = OptPair_MaxSharpe(look_back_start=look_back_start, curr_date=curr_date, look_back_end=look_back_end,
                                     validate_end=validate_end,
                                     run_pairs=run_pairs,
                                     config_file_yaml=config_file_yaml0,
                                     n_jobs=1, n_trials=100)

        # 执行优化，并保存最优参数结果
        testopt1.opt_dump(best_arg_path=best_arg_path, result_filename=result_filename, direction='maximize')

        testopt1.validate_result(best_arg_path=best_arg_path, result_filename=result_filename)

        if testopt1._validate_result is not None and not testopt1._validate_result.empty:
            validate_result_top = testopt1._validate_result.sort_values('value', ascending=False).head(n=1)

            for _, row in validate_result_top.iterrows():
                volLookback = row['volLookback']
                s_period = row['s_period']
                l_period = row['l_period']

                dailyreturns = testopt1.get_dailyreturns(volLookback=volLookback, s_period=s_period, l_period=l_period)
                sub_dailyreturns = Analysis_func.cut_returns(daily_returns=dailyreturns, look_back_start=validate_end,
                                                             look_back_end=next_opt_date)
                sub_dailyreturns_list.append(sub_dailyreturns)
    dailyreturns = pandas.concat(sub_dailyreturns_list)

    result_filename = 'opt_backtest_dailyreturns_' + '_'.join(run_pairs) + '.csv'
    dailyreturns.to_csv(os.path.join(opt_result_path, result_filename))
    cache.clear_cache()

    return os.path.join(opt_result_path, result_filename)


def run_benchmark(each_pair0,config_file_yaml0,best_arg_path,re_calc=True):
    DataFactory.config(MONGDB_USER='zhangfang', MONGDB_PW='jz148923', DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE
                       ,logging_level=global_variable.logging.INFO
                       ,temp_path=r'E:\TEMP_Python\MOMENTUM'
                       )
    run_pairs = [each_pair0]
    namespace = 'CtaMomentum_'.join(run_pairs)
    cache.load(namespace)
    benchmark_result_path = os.path.join(best_arg_path,'benchmark_result')
    check_fold(benchmark_result_path)

    volLookback = 3
    speeds = 6
    scaler = 3
    result_filename = 'benchmark_arg_series' + '_'.join(run_pairs) + '.csv'
    curr_date = datetime.date.today().strftime('%Y-%m-%d')
    testopt1 = OptPair_MaxSharpe(look_back_start=None, look_back_end=None, validate_end=None,curr_date=curr_date,
                                 run_pairs=run_pairs,
                                 config_file_yaml=config_file_yaml0,
                                 n_jobs=1, n_trials=100)

    testopt1.get_strategy_obj()
    daily_return_by_init_aum = testopt1.get_dailyreturns(volLookback,speeds,scaler,re_calc=re_calc)
    daily_return_by_init_aum.to_csv(os.path.join(benchmark_result_path, result_filename))

    return os.path.join(benchmark_result_path, result_filename)


if __name__ == '__main__':
    DataFactory.config(MONGDB_USER='zhangfang', MONGDB_PW='jz148923', DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE
                       , logging_level=global_variable.logging.INFO
                       , temp_path=r'E:\TEMP_Python\MOMENTUM')

    print(cache.cache_path())

    backtest_path = r'E:/Strategy/MOMENTUM'

    config_file_yaml = r'C:/Users/Administrator/PycharmProjects/resFuturesCTA_momentum/backtest_config_cta_momentum/config_sample_1d.yaml'

    # 优化参数配置文件保存到指定路径供回测使用
    best_arg_path = os.path.join(backtest_path, r'best_args_maxsharp')
    report_path = os.path.join(backtest_path, r'report')

    check_fold(best_arg_path)
    check_fold(report_path)
    # 回测盘
    config = Config_back_test().load(config_file=config_file_yaml)
    products = config.get_strategy_config('product_group')
    run_pairs_0 = ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM']
    run_pairs_0 = ['A', 'AG', 'AL', 'AP', 'AU', 'B', 'BU', 'C', 'CF', 'CS', 'CU', 'FG',
                   'HC', 'I', 'IC', 'IF', 'IH', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P',
                   'PB', 'PP', 'RB', 'RM', 'RU', 'SC', 'SF', 'SM', 'SN', 'SR',
                   'T', 'TA', 'TF', 'V', 'Y', 'ZC', 'ZN']  # 所有品种
    # run_pairs_0 = ['TA']

    pool = Pool(5)
    ret_list = []
    # 品种，和起始年份（因为各品种起始日期不一，单独控制一下）
    for each_pair in run_pairs_0:
        ret_list.append(pool.apply_async(run, args=(each_pair, config_file_yaml, best_arg_path)))
        # opt_backtest_pathfile = run(each_pair0=each_pair,config_file_yaml0=config_file_yaml,best_arg_path=best_arg_path)
    pool.close()
    pool.join()
    # opt_result_path = os.path.join(best_arg_path, 'opt_result')  # 各品种优化参数的回测结果文件夹
    # format_folder_func = lambda x: x.replace('opt_backtest_dailyreturns_', '').replace('.csv', '').split('_')[
    #     0]  # 从文件名提取品种标的， 可以实例化Future的代码
    # strategy_name = 'resFuturesCtaMomentum'
    # filename = None  # 不过滤文件，全部提取
    # report_strategy_summary(strategy_name=strategy_name, format_folder_func=format_folder_func, path=opt_result_path,
    #                         filename=filename
    #                         , pdf_path=report_path)
    # report_strategy_sharpelist(strategy_name=strategy_name, format_folder_func=format_folder_func, path=opt_result_path,
    #                            filename=filename
    #                            , pdf_path=report_path)
