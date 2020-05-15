#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/2/24 15:13
# @Author  : jwliu
# @Site    : 
# @Software: PyCharm
import sys
import os

CurrentPath = os.path.dirname(__file__)
# print(CurrentPath)
sys.path.append(CurrentPath.replace('opt', ''))
import datetime
import pandas
import pytz
from common.os_func import check_fold
from common.cache import cache
from multiprocessing import Pool,cpu_count
import optuna
from opt.base_opt import BaseOpt
from config.config import Config_back_test
from data_engine.data_factory import DataFactory
import data_engine.global_variable as global_variable
from ctamomentum1 import CtaCtaStrategy_ex
from execution.execution import Execution_ex
from execution.execution_v2 import Execution_ex as Execution_ex_v2
from settlement.settlement import Settlement_ex
from analysis.analysis import Analysis_func
from analysis.execution_settle_analysis import execution_settle_analysis
import copy


class OptPair_MaxSharpe(BaseOpt):
    '''
        最大sharpe
    '''

    def __init__(self, config_file_yaml, curr_date, run_pairs, look_back_start, look_back_end
                 , n_trials=100, n_jobs=-1):
        BaseOpt.__init__(self, look_back_start=look_back_start, look_back_end=look_back_end, n_trials=n_trials,
                         n_jobs=n_jobs)
        self._config_file_yaml = config_file_yaml
        self._config = Config_back_test(config_path=os.path.split(__file__)[0]).load(config_file=self._config_file_yaml)
        _strategy_obj = cache.get_object('CtaMomentumStrategy', namespace='_strategy_obj')
        if _strategy_obj is not None:
            self._strategy_obj = _strategy_obj
        else:
            self._strategy_obj = CtaCtaStrategy_ex(config=self._config, curr_date=curr_date, symbol_list=run_pairs,
                                                asset_type=global_variable.ASSETTYPE_FUTURE)
            self._strategy_obj.load_data(startDate=None, endDate=None)
            cache.set_object('CtaMomentumStrategy', self._strategy_obj, namespace='_strategy_obj')

    def objective(self, trial):
        '''
        optuna 的迭代函数，objective函数名不要修改； 也可以是独立的函数；
        :param trial:
        :return:
        '''
        # 优化的自变量
        volLookback = trial.suggest_int('volLookback', 1, 6)  # 在20，120 之间的（整数）离散区间优化
        s_period = trial.suggest_int('s_period', 1, 45)
        l_period = trial.suggest_int('l_period', 1, 45)

        # 查询缓存相同参数下的daily_return结果
        daily_return = self.get_cache(volLookback, s_period, l_period)
        if daily_return is None:  # 缓存无结果
            print('back test', 20*volLookback, 3*s_period, 3*l_period + 3*s_period)

        # 执行回测
            _config = copy.copy(self._config)
            _config.strategy_config['volLookback'] = volLookback * 20
            _config.strategy_config['s_period'] = [3*s_period]
            _config.strategy_config['l_period'] = [3*l_period + 3*s_period]
            _config.strategy_id = None
            signal_dataframe = self._strategy_obj.run_test(config=_config
                                                           , s_period=_config.strategy_config['s_period']
                                                           , l_period=_config.strategy_config['l_period']
                                                           , volLookback=_config.strategy_config['volLookback'])
            (success, positions_dataframe) = Execution_ex(config=_config).exec_trading(
                signal_dataframe=signal_dataframe)

            if success:
                settlement_obj = Settlement_ex(config=_config)
                settlement_obj.settle(positions_dataframe=positions_dataframe)
                daily_return = settlement_obj.daily_return

                #写入缓存
                self.cache_result(settlement_obj.daily_return, volLookback, s_period, l_period)
        else:
            # 缓存有结果
            print('is cache result', volLookback, s_period, l_period)
        if daily_return is None:
            return -1
            # 截取_look_back_start到_look_back_end之间的daily return
        if isinstance(daily_return, pandas.Series):
            daily_return_ = daily_return
            if self._look_back_start is not None:
                daily_return_ = daily_return_[daily_return_.index >= (self._look_back_start + datetime.timedelta(days=1))]  # _look_back_start到_look_back_end 前开后闭
            if self._look_back_end is not None:
                daily_return_ = daily_return_[daily_return_.index <= self._look_back_end]

            # 返回目标值（当前为回测的夏普值）
            return Analysis_func.sharpe_ratio(daily_returns=daily_return_)
        # 返回目标值（当前为回测的夏普值）
        return -1


class OptPair_MinMaxDrawDown(OptPair_MaxSharpe):
    '''
        以最小化 最大回测 为目标
        继承了OptPair_MaxSharpe，修改objective的返回目标值
    '''

    def objective(self, trial):
        # 优化的自变量
        volLookback = trial.suggest_int('volLookback', 20, 120)
        speeds = trial.suggest_int('speeds', 5, 9)
        scaler = trial.suggest_int('scaler', 2, 4)

        # 执行回测
        _config = copy.copy(self._config)
        _config.strategy_config['volLookback'] = volLookback
        _config.strategy_config['speeds'] = [speeds, speeds + 1, speeds + 2]
        _config.strategy_config['scaler'] = scaler
        _config.strategy_id = None
        signal_dataframe = self._strategy_obj.run_test(startDate=self._look_back_start
                                                       , endDate=self._look_back_end
                                                       , config=_config
                                                       , scaler=_config.strategy_config['scaler']
                                                       , speeds=_config.strategy_config['speeds']
                                                       , volLookback=_config.strategy_config['volLookback'])
        # print(signal_dataframe.index[0], signal_dataframe.index[-1])
        analysis_obj = execution_settle_analysis(signal_dataframe=signal_dataframe, config=self._config)

        return - analysis_obj.max_drawdown()


def run(each_pair0, config_file_yaml0):
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE,logging_level=global_variable.logging.INFO)

    config = Config_back_test().load(config_file=config_file_yaml0)
    #opt_date_list 每个月1日作为优化参数点，产生优化日的列表
    run_pairs = [each_pair0]
    curr_date = datetime.date.today().strftime('%Y-%m-%d')
    strategy_obj = CtaCtaStrategy_ex(config=config, curr_date=curr_date, symbol_list=run_pairs,
                                      asset_type=global_variable.ASSETTYPE_FUTURE)
    strategy_obj.load_data(startDate=None, endDate=None)
    cache.set_object('CtaMomentumStrategy', strategy_obj, namespace='_strategy_obj')
    start_date = strategy_obj.start_date

    # opt_date_list 每个月1日作为优化参数点，产生优化日的列表
    opt_date_list = []
    for yr in range(2000, 2021):
        for mt in range(1, 13):
            dt = datetime.datetime(yr, mt, 1, 15, 0, 0, 0)
            dt2 = datetime.datetime(yr, mt, 1, 15, 0, 0, 0, tzinfo=pytz.timezone('PRC'))
            if dt > datetime.datetime.now():
                continue
            elif dt2 < start_date + datetime.timedelta(days=365):
                continue
            opt_date_list.append(dt2)
    # print(opt_date_list)
    best_arg_path = r'E:/Strategy/MOMENTUM/best_args'
    check_fold(best_arg_path)

    ret_list = []
    for look_back_end in opt_date_list:
        look_back_start = look_back_end + datetime.timedelta(days=-365)

        # 优化参数保存文件名
        result_filename = 'best_arg_series' + '_'.join(run_pairs) + '_' + look_back_end.strftime(
            '%Y%m%d') + '.csv'
        if os.path.exists(os.path.join(best_arg_path, result_filename)):
            continue

        # n_trials，
        testopt1 = OptPair_MaxSharpe(curr_date=look_back_end.strftime('%Y-%m-%d'), look_back_start=look_back_start,
                                     look_back_end=look_back_end,
                                     run_pairs=run_pairs,
                                     config_file_yaml=config_file_yaml0,
                                     n_jobs=1, n_trials=100)
        # 执行优化，并保存最优参数结果
        testopt1.opt_dump(best_arg_path=best_arg_path
                          , result_filename=result_filename
                          , direction='maximize')
    cache.clear_cache()


if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE,
                       logging_level=global_variable.logging.INFO)
    pool = Pool(5)
    config_file_yaml = r'C:/Users/Administrator/PycharmProjects/resFuturesCTA_momentum/backtest_config_cta_momentum/config_sample_1d.yaml'
    # 回测盘
    config = Config_back_test().load(config_file=config_file_yaml)

    products = config.get_strategy_config('product_group')
    run_pairs_0 = [tuple([y for y in x]) for x in products]

    run_pairs_0 = ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM']
    run_pairs_0 = ['A', 'AG', 'AL', 'AP', 'AU', 'B', 'BU', 'C', 'CF', 'CS', 'CU', 'FG',
                   'HC', 'I', 'IC', 'IF', 'IH', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P',
                   'PB', 'PP', 'RB', 'RM', 'RU', 'SC', 'SF', 'SM', 'SN', 'SR',
                   'T', 'TA', 'TF', 'V', 'Y', 'ZC', 'ZN']  # 所有品种
    # run_pairs_0 = ['TA']

    for each_pair in run_pairs_0:
        pool.apply_async(run, args=(each_pair, config_file_yaml))
        # run(each_pair0=each_pair,config_file_yaml0=config_file_yaml)
    pool.close()
    pool.join()


    # import pandas
    # import pytz
    # from common.os_func import check_fold
    # from multiprocessing import Pool, cpu_count
    # curr_date = datetime.date.today().strftime('%Y-%m-%d')
    #
    # DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE,
    #                    logging_level=global_variable.logging.INFO)
    # n_jobs = 5  # 默认-1指使用全部cpu进行优化迭代
    # config_file_yaml = r'C:/Users/Administrator/PycharmProjects/resFuturesCTA_YMJH/backtest_config_cta_ymjh/config_sample_1d.yaml'
    # config = Config_back_test().load(config_file=config_file_yaml)
    # products = config.get_strategy_config('product_group')
    # run_pairs_0 = [tuple([y for y in x]) for x in products]
    # run_pairs_0 = ['TA']
    # # run_pairs_0 = ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM']
    #
    # # 品种，和起始年份（因为各品种起始日期不一，单独控制一下）
    # for each_pair in run_pairs_0:
    #     run_pairs = [each_pair]
    #     strategy_obj = CtaYmjhStrategy_ex(config=config, curr_date=curr_date, symbol_list=run_pairs,
    #                                             asset_type=global_variable.ASSETTYPE_FUTURE)
    #     strategy_obj.load_data(startDate=None, endDate=None)
    #     start_date = strategy_obj.start_date
    #
    #     # opt_date_list 每个月1日作为优化参数点，产生优化日的列表
    #     opt_date_list = []
    #     for yr in range(2000, 2021):
    #         for mt in range(1, 13):
    #             dt = datetime.datetime(yr, mt, 1, 15, 0, 0, 0)
    #             dt2 = datetime.datetime(yr, mt, 1, 15, 0, 0, 0, tzinfo=pytz.timezone('PRC'))
    #             if dt > datetime.datetime.now():
    #                 continue
    #             elif dt2 < start_date + datetime.timedelta(days=365):
    #                 continue
    #             opt_date_list.append(dt2)
    #     print(opt_date_list)
    #
    #     # 优化参数配置文件保存到指定路径供回测使用
    #     best_arg_path = r'E:/Strategy/YMJH/best_args'
    #     check_fold(best_arg_path)
    #
    #     ret_list = []
    #     for look_back_end in opt_date_list:
    #         look_back_start = look_back_end + datetime.timedelta(days=-365)
    #
    #         # 优化参数保存文件名
    #         result_filename = 'best_arg_series' + '_'.join(run_pairs) + '_' + look_back_end.strftime(
    #             '%Y%m%d') + '.csv'
    #         if os.path.exists(os.path.join(best_arg_path, result_filename)):
    #             continue
    #
    #
    #         # n_trials，
    #         testopt1 = OptPair_MaxSharpe(curr_date=look_back_end.strftime('%Y-%m-%d'), look_back_start=look_back_start, look_back_end=look_back_end,
    #                                      run_pairs=run_pairs,
    #                                      config_file_yaml=config_file_yaml,
    #                                      n_jobs=n_jobs, n_trials=100)
    #         # 执行优化，并保存最优参数结果
    #         testopt1.opt_dump(best_arg_path=best_arg_path
    #                           , result_filename=result_filename
    #                           , direction='maximize')
