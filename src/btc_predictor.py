import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir,'btcusd_1-min_data.csv')

df = pd.read_csv(file_path)


# print("Loading Bitcoin Data....")
# df = pd.read_csv('btcusd_1-min_data.csv')

df['Date'] = pd.to_datetime(df['Timestamp'],unit='s')
print("\nTimestamp converted. Sample:")
print(df[['Timestamp','Date']].head(3))

df.set_index('Date',inplace=True)
df.drop('Timestamp',axis = 1 , inplace = True)

df_daily = df.resample('D').agg({
    'Open' : 'first',
    'High' : 'max',
    'Low' : 'min',
    'Close':'last',
    'Volume':'sum'
})

print("\n Misising values before cleaning",df_daily.isnull().sum())
df_daily.dropna(inplace = True)
print("Missing values after cleaning\n",df_daily.isnull().sum())

print("\nClean daily shape:",df_daily.shape)
print("\nFirst 5 rows:\n",df_daily.head())
print("\n last 5 rows:\n",df_daily.tail())