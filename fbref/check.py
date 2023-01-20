import pandas as pd

df = pd.read_csv('v2/data/all_combined.csv')

df = df.dropna()

print(df.head())