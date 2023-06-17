import pandas as pd


df = pd.read_csv('bbw.csv')


bbw_average = df['Bollinger Bands Width'].mean()

print(f"Bollinger BandWidth의 평균은 {bbw_average}입니다.")
