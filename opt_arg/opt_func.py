import pandas
import datetime
import pytz
import data_engine.global_variable as global_variable
def get_opt_date_df(start_date,dropna = True):
    opt_date_list = []
    for yr in range(2000,2021):
        for mt in range(1,13):
            dt = datetime.datetime(yr,mt,1,15,0,0,0)
            dt2 = datetime.datetime(yr,mt,1,15,0,0,0).astimezone(pytz.timezone(global_variable.DEFAULT_TIMEZONE))
            if dt > datetime.datetime.now():
                continue
            elif dt2 < start_date + datetime.timedelta(days=365):
                continue
            opt_date_list.append(dt2)
    opt_date_series = pandas.Series(opt_date_list,index=range(len(opt_date_list)))
    opt_date_df = pandas.concat([opt_date_series,opt_date_series.shift(1),opt_date_series.shift(-1)],axis=1)
    opt_date_df.columns = ['opt_date','last_opt_date','next_opt_date']
    opt_date_df['next_opt_date'] = opt_date_df['next_opt_date'].fillna(datetime.datetime.now().astimezone(pytz.timezone(global_variable.DEFAULT_TIMEZONE)))
    if dropna:
        opt_date_df = opt_date_df.dropna(subset=['opt_date','last_opt_date'])
    return opt_date_df