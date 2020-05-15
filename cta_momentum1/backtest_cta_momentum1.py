# -*- coding: utf-8 -*-
import sys
import os
CurrentPath = os.path.dirname(__file__)
print(CurrentPath)
sys.path.append(CurrentPath.replace('cta_momentum', ''))
print(sys.path)
import datetime
import warnings
import time
warnings.filterwarnings("ignore")
import traceback
import pandas
from execution.execution import Execution
from analysis.analysis import Analysis
from cta_momentum1.ctamomentum1 import CtaMomentumStrategy
from settlement.settlement import Settlement
from data_engine.data_factory import DataFactory
import data_engine.setting as setting

from common.file_saver import file_saver
from common.os_func import check_fold
from data_engine.setting import ASSETTYPE_FUTURE, FREQ_1M, FREQ_5M, FREQ_1D
# DataFactory.sync_future_from_remote()


def run_backtest(run_symbols, freq, result_folder, strategy_params, run_params, exec_lag, saving_file=False):
    import time
    t1 = time.clock()
    try:
        DataFactory.config(MONGDB_PW='jz2018*',DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
        # 策略对象
        strategy_obj = CtaMomentumStrategy(symbols_list=run_symbols,
                                      freq=freq,
                                      asset_type=ASSETTYPE_FUTURE,
                                      result_fold=result_folder,
                                      **strategy_params
                                     )

        # 回测
        signal_dataframe = strategy_obj.run_test(startDate=run_params['start_date'], endDate=run_params['end_date'],
                                                 **run_params
                                                 )

        execution_obj = Execution(freq=freq, exec_price_mode=Execution.EXEC_BY_OPEN, exec_lag=exec_lag)
        (success, positions_dataframe) = execution_obj.exec_trading(signal_dataframe=signal_dataframe)

        if not success:
            print(positions_dataframe)
            assert False

        if success:
            settlement_obj = Settlement(init_aum=run_params['capital'])
            # file_saver().save_file(positions_dataframe, os.path.join(result_folder, 'positions_dataframe.csv'))
            settlement_obj.settle(positions_dataframe=positions_dataframe)
            print(settlement_obj.daily_return)

            # 分析引擎，  结果保存到result_folder文件夹下
            analysis_obj = Analysis(daily_returns=settlement_obj.daily_return_by_init_aum,
                                    daily_positions=settlement_obj.daily_positions,
                                    daily_pnl=settlement_obj.daily_pnl,
                                    daily_pnl_gross=settlement_obj.daily_pnl_gross,
                                    daily_pnl_fee=settlement_obj.daily_pnl_fee,
                                    transactions=settlement_obj.transactions,
                                    round_trips=settlement_obj.round_trips,
                                    result_folder=result_folder,
                                    strategy_id='_'.join([strategy_obj._strategy_name, '_'.join(strategy_obj._symbols)]),
                                    symbols=strategy_obj._symbols,
                                    strategy_type=strategy_obj._strategy_name)
            # analysis_obj.plot_cumsum_pnl(show=False,title='_'.join([strategy_obj._strategy_name, '_'.join(strategy_obj._symbols)]))
            # analysis_obj.plot_all()
            analysis_obj.save_result()
    except:
        if saving_file:
            file_saver().join()
        traceback.print_exc()
    if saving_file:
        file_saver().join()
    print('=================','run_backtest','%.6fs' % (time.clock()-t1))


def PowerSetsRecursive(items):
    # 求集合的所有子集
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result


if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_LOCAL)
    client = DataFactory.get_mongo_client()
    print(client.database_names())
    from multiprocessing import Pool, cpu_count
    pool = Pool(max(1, cpu_count() - 4))
    # pool = Pool(1)

    file_save_obj = file_saver()
    # symbols = ['NI', 'SM', 'SR', 'L', 'BU', 'MA', 'AG', 'CF']
    symbols = ['SM', 'NI', 'J', 'I', 'RB', 'HC', 'IF', 'AP', 'TA', 'SR']
    symbols = ['A', 'AG', 'AL', 'AP', 'AU', 'BU', 'C', 'CF', 'CJ', 'CU', 'EG', 'FG', 'HC', 'I', 'IC', 'IF',
               'IH', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P', 'PP', 'RB', 'RM', 'RU', 'SC', 'SM', 'SN',
               'SP', 'SR', 'T', 'TA', 'TF', 'TS', 'WH', 'Y', 'ZC', 'ZN']
    symbols = ['A', 'AG', 'AL', 'AP', 'AU', 'B', 'BU', 'C', 'CF', 'CS', 'CU', 'CY', 'FG', 'HC', 'I', 'IC', 'IF', 'IH',
               'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P', 'PB', 'PP', 'RB', 'RM', 'RU', 'SC', 'SF', 'SM', 'SN',
               'SR', 'T', 'TA', 'TF', 'V', 'Y', 'ZC', 'ZN']
    # symbols = ['TF']

    symbols = [i + '_VOL' for i in symbols]
    symbols_dict = {'Grains': ['C', 'CS', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM'],
                    'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU'],
                    'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],
                    'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM'],
                    'Equity': ['IF', 'IH', 'IC'],
                    'Bonds': ['T', 'TF'],
                    'PreciousMetal': ['AG', 'AU']}
    s_period_lst = [i for i in range(3, 131, 3)]
    l_period_lst = [i for i in range(6, 262, 6)]
    class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal']
    for clas in class_lst:
        symbols = symbols_dict[clas]
        symbols = [i + '_VOL' for i in symbols]
        run_symbols_0 = [symbols]
        for s_period in s_period_lst:
            for l_period in l_period_lst:
                # l_period = 2 * s_period
                if s_period >= l_period:
                    continue
                for each in run_symbols_0:

                    # each.extend([i + '_VOL' for i in ['J', 'I', 'RB', 'HC', 'IF', 'AP', 'TA', 'SR']])
                    run_symbols = each
                    print(each)
                    run_params = {'capital': 100000000,
                                  'daily_start_time': '9:00:00',
                                  'daily_end_time': '23:30:00',
                                  'start_date': '20100101',
                                  'end_date': '20200630'
                                  }

                    strategy_params = {'period': '1d',
                                       's_period_lst': [s_period],
                                       'l_period_lst': [l_period],
                                       'SW': 63,
                                       'PW': 252,
                                       'maxLeverage': 4,
                                       'targetVol': 0.1,
                                       }
                    exec_lag = 1
                    result_folder = 'e://Strategy//MOMENTUM//better//resRepo_momentum_%s_%s_%s' % (clas, s_period, l_period)
                    check_fold(result_folder)
                    freq = FREQ_1M
                    if strategy_params['period'] == 5:
                        freq = FREQ_5M
                    elif strategy_params['period'] == 1:
                        freq = FREQ_1M
                    else:
                        freq = FREQ_1D
                    try:
                        # 策略对象
                        print('resRepo_%s_%s_%s' % (clas, s_period, l_period))
                        pool.apply_async(run_backtest,
                                         args=(run_symbols, freq, result_folder, strategy_params, run_params, exec_lag, True))
                        # run_backtest(run_pairs, freq, result_folder, strategy_params, run_params, exec_lag,saving_file=False)
                        DataFactory().clear_data()
                    except:
                        DataFactory().clear_data()
                        traceback.print_exc()
    pool.close()
    file_saver().join()
    pool.join()
