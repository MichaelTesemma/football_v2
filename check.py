import pandas as pd
from os import listdir

data = pd.read_csv('v2/prem_clean_fixtures_and_dataframes/2019_2020_2021_2022_premier_league_fixtures_df.csv')
print(data['Fixture ID'])

dir = listdir('v2/prem_game_stats_json_files')
print(dir)
li = []
for i in dir:
    li.append(int(i[:-5]))
droppping_list = []
print(len(li), len(data['Fixture ID']))
for i in data['Fixture ID']:
    if i not in li:
        droppping_list.append(i)


df = data.drop(data[data['Fixture ID'].isin(droppping_list)].index)
print(len(df['Fixture ID']))

for i in df['Fixture ID']:
    if i not in droppping_list:
        print(i)