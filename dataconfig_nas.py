import pandas as pd


df = pd.read_csv('nasdaq15.csv')


df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

start_date = pd.to_datetime('2022-07-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
end_date = pd.to_datetime('2023-02-26 00:00:00', format='%Y-%m-%d %H:%M:%S')

filtered_df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

filtered_df.to_csv('nas_learn_7_2.csv', index=False)