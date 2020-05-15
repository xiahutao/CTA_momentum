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
sys.path.append(CurrentPath.replace('opt_arg', ''))
import datetime

import pandas
import pytz
from common.os_func import check_fold
from common.cache import cache
from multiprocessing import Pool, cpu_count

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
from opt_arg.opt_func import get_opt_date_df


class OptPair_MaxSharpe(BaseOpt):
    '''
        最大sharpe
    '''
    def __init__(self, config_file_yaml, curr_date, run_pairs
                 , look_back_start, look_back_end
                 , validate_end=None
                 , n_trials=100, n_jobs=-1
                 , cache_namespace=None):
        if cache_namespace is None:
            cache_namespace = 'CtaMomentum_'.join(run_pairs)
        BaseOpt.__init__(self, look_back_start=look_back_start, look_back_end=look_back_end
                         , n_trials=n_trials, n_jobs=n_jobs
                         , cache_namespace=cache_namespace
                         , load_cache=True)

        self._validate_end = validate_end
        self._validate_result = None
        self._run_pairs = run_pairs
        self._curr_date = curr_date
        self._config_file_yaml = config_file_yaml
        self._config = Config_back_test(config_path=os.path.split(__file__)[0]).load(config_file=self._config_file_yaml)

        # 策略对象缓存
        namespace = '_'.join(['CtaMomentumStrategy'] + list(self._run_pairs))
        if not cache.load(namespace=namespace):
            self._strategy_obj = CtaCtaStrategy_ex(config=self._config, curr_date=self._curr_date, symbol_list=self._run_pairs,
                                                asset_type=global_variable.ASSETTYPE_FUTURE)
            self._strategy_obj.load_data(startDate=None, endDate=None)
            cache.set_object('strategy_obj', self._strategy_obj, namespace=namespace)
        else:
            self._strategy_obj = cache.get_object('strategy_obj', namespace=namespace)

    def validate_result(self, best_arg_path, result_filename):
        if self._study is None or self._validate_end is None:
            return None
        trials_dataframe = self._study.trials_dataframe()
        params_cols = [x for x in trials_dataframe.columns if 'params_' in x]
        trials_dataframe = trials_dataframe.sort_values('value', ascending=False).drop_duplicates(subset=params_cols)
        keep_heads = min(1, int(len(trials_dataframe) * 0.1))
        result_list = []
        for _, row in trials_dataframe.head(n=keep_heads).iterrows():
            volLookback = row['params_volLookback']
            s_period = row['params_s_period']
            l_period = row['params_l_period']

            daily_return_by_init_aum = self.get_dailyreturns(volLookback=volLookback,
                                                             s_period=s_period,
                                                             l_period=l_period)
            objective_value = self.get_objective_value(daily_returns=daily_return_by_init_aum,
                                                       fromdate=self._look_back_end, todate=self._validate_end)

            result = pandas.Series(
                {'volLookback': volLookback, 's_period': s_period, 'l_period': l_period, 'value': objective_value})
            result_list.append(result)
        result_df = pandas.concat(result_list, axis=1).T
        result_df.sort_values('value', ascending=False).to_csv(
            os.path.join(best_arg_path, 'validate_result_' + result_filename))
        self._validate_result = result_df
        return result_df

    def get_dailyreturns(self, volLookback, s_period, l_period, re_calc=False):
        daily_return_by_init_aum = None
        if not re_calc:
            daily_return_by_init_aum = cache.get_object((volLookback, s_period, l_period),
                                                        namespace=self._cache_namespace)

        if daily_return_by_init_aum is None:  # 缓存无结果
            print('back test', volLookback, s_period, l_period)
            # 执行回测（全历史回测）
            _config = copy.copy(self._config)
            _config.strategy_config['volLookback'] = volLookback * 20
            _config.strategy_config['s_period'] = [3*s_period]
            _config.strategy_config['l_period'] = [3*l_period + 3*s_period]
            _config.strategy_id = None
            signal_dataframe = self._strategy_obj.run_test(config=_config
                                                           , s_period=_config.strategy_config['s_period']
                                                           , l_period=_config.strategy_config['l_period']
                                                           , volLookback=_config.strategy_config['volLookback'])
            settlement_obj = self._strategy_obj.get_settle_obj(config=_config, signal_dataframe=signal_dataframe)
            if settlement_obj is not None:
                daily_return_by_init_aum = settlement_obj.daily_return_by_init_aum

                # 写入缓存
                cache.set_object((volLookback, s_period, l_period)
                                 , settlement_obj.daily_return_by_init_aum
                                 , namespace=self._cache_namespace)
                cache.dump(namespace=self._cache_namespace)
            return daily_return_by_init_aum
        else:  # 缓存有结果
            print('is cache result', volLookback, s_period, l_period)
            return daily_return_by_init_aum

    def get_objective_value(self, daily_returns, fromdate=None, todate=None):
        if daily_returns is None:
            print(self._cache_namespace)
        return Analysis_func.sharpe_ratio(daily_returns=daily_returns, look_back_start=fromdate, look_back_end=todate)

    def objective(self, trial):
        '''
        optuna 的迭代函数，objective函数名不要修改； 也可以是独立的函数；
        :param trial:
        :return:
        '''

        # 优化的自变量
        # volLookback = trial.suggest_uniform('volLookback',20,120)  # 在20 120的连续区间优化
        volLookback = trial.suggest_int('volLookback', 1, 6)  # 在20，120 之间的（整数）离散区间优化
        s_period = trial.suggest_int('s_period', 1, 45)
        l_period = trial.suggest_int('l_period', 1, 45)

        # 查询缓存相同参数下的daily_return结果
        daily_return_by_init_aum = self.get_dailyreturns(volLookback=volLookback, s_period=s_period, l_period=l_period)
        if daily_return_by_init_aum is None:
            return -1

        # 截取_look_back_start到_look_back_end之间的daily return
        if isinstance(daily_return_by_init_aum, pandas.Series):
            # 返回目标值（当前为回测的夏普值）
            return self.get_objective_value(daily_returns=daily_return_by_init_aum, fromdate=self._look_back_start,
                                            todate=self._look_back_end)
        # 返回目标值（当前为回测的夏普值）
        return -1
