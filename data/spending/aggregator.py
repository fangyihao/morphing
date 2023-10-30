import pandas as pd
import os
import numpy as np
def group_iTrade_SEM2_Daily():
    df = pd.read_csv('processed/iTrade_SEM2_Daily.csv', parse_dates=['Date'])
    print(df.head())

    df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(6, unit='d')
    df = df.groupby([pd.Grouper(key='Date', freq='W-SUN')])['Spend'].sum().reset_index().sort_values('Date')
    df.rename(columns = {'Date':'Week'}, inplace = True)
    print (df)

    df.to_csv('processed/iTrade_SEM2_Weekly.csv', index=False)

def group_Wealth_SEM_Daily():
    df = pd.read_csv('processed/Wealth_SEM_Daily.csv', parse_dates=['Date'])
    print(df.head())

    df['Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(6, unit='d')
    df = df.groupby([pd.Grouper(key='Date', freq='W-SUN')])['Spend'].sum().reset_index().sort_values('Date')
    df.rename(columns = {'Date':'Week'}, inplace = True)
    print (df)

    df.to_csv('processed/Wealth_SEM_Weekly.csv', index=False)
    
def consolidate_spending():
    business_units = ["iTrade", "Wealth"]
    channels = ["SEM", "SEM2", "Digital", "DigitalOOH", "Radio", "Social", "TV", "OOH", "Print"]
    dfs = []
    for business_unit in business_units:
        for channel in channels:
            csv_file = "processed/{}_{}_Weekly.csv".format(business_unit, channel)
            if os.path.isfile(csv_file):
                print(csv_file)
                df = pd.read_csv(csv_file, parse_dates=['Week'])
                df['Week'] = pd.to_datetime(df['Week']) - pd.to_timedelta(6, unit='d')
                df = df.groupby([pd.Grouper(key='Week', freq='W-SUN')])['Spend'].sum().reset_index().sort_values('Week')
                df['Business Unit'] = business_unit
                df['Channel'] = channel
                df = df[['Week', 'Business Unit', 'Channel', 'Spend']]
                dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    dfs.to_csv('processed/iTrade_Wealth_Weekly.csv', index=False)
    
def pivot_spending():
    df = pd.read_csv('processed/iTrade_Wealth_Weekly.csv', parse_dates=['Week'])
    table = pd.pivot_table(df, values='Spend', index=['Week', 'Business Unit'], columns=['Channel'], aggfunc=np.sum, fill_value=0)
    table.to_csv('processed/iTrade_Wealth_PivotTable_Weekly.csv', index=True)
    
group_iTrade_SEM2_Daily()
group_Wealth_SEM_Daily()

consolidate_spending()
pivot_spending()