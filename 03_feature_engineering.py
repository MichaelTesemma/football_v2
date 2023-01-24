# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:23:13 2020

@author: mhayt
"""


print('\n\n ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import time
start=time.time()

import pickle
from ml_functions.feature_engineering_functions import average_stats_df
from ml_functions.feature_engineering_functions import creating_ml_df


#------------------------------- INPUT VARIABLES ------------------------------

#Please state the name of the saved nested dictionary generated with '02_cleaning_stats_data.py', as well as the name of the saved output files (stats DataFrame).

stats_dict_saved_name = '2019_2020_2021_2022_prem_all_stats_dict.txt'

df_5_output_name = 'prem_df_for_ml_5_v2.txt'
df_10_output_name = 'prem_df_for_ml_10_v2.txt'
df_3_output_name = 'prem_df_for_ml_3_v2.txt'
df_2_output_name = 'prem_df_for_ml_2_v2.txt'
df_4_output_name = 'prem_df_for_ml_4_v2.txt'
df_1_output_name = 'prem_df_for_ml_1_v2.txt'
df_6_output_name = 'prem_df_for_ml_6_v2.txt'
df_7_output_name = 'prem_df_for_ml_7_v2.txt'
df_8_output_name = 'prem_df_for_ml_8_v2.txt'
df_9_output_name = 'prem_df_for_ml_9_v2.txt'






#----------------------------- FEATURE ENGINEERING ----------------------------

with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{stats_dict_saved_name}', 'rb') as myFile:
    game_stats = pickle.load(myFile)

#creating a list with the team id in
team_list = []
for key in game_stats.keys():
    team_list.append(key)
team_list.sort()

#creating a dictionary with the team id as key and fixture id's as values
team_fixture_id_dict = {}
for team in team_list:
    fix_id_list = []
    for key in game_stats[team].keys():
        fix_id_list.append(key)
    fix_id_list.sort()
    sub_dict = {team:fix_id_list}
    team_fixture_id_dict.update(sub_dict)
        
#the list of fixtures was first home then away, we want them in chronological order so we need to sort them.
for team in team_fixture_id_dict:
    team_fixture_id_dict[team].sort()

#we can now iterate over the fixture ID list given a team id key using the dict created above. N.B./ the number of games over which the past data is averaged. A large number will smooth out past performance where as a small number will result in the prediction being heavily reliant on very recent form. This is worth testing the ml model build phase.

#5 game sliding average.
df_ml_5 = average_stats_df(5, team_list, team_fixture_id_dict, game_stats)

#10 game sliding average.
df_ml_10 = average_stats_df(10, team_list, team_fixture_id_dict, game_stats)

#3 game sliding average.
df_ml_3 = average_stats_df(3, team_list, team_fixture_id_dict, game_stats)

#2 game sliding average.
df_ml_2 = average_stats_df(2, team_list, team_fixture_id_dict, game_stats)

#2 game sliding average.
df_ml_4 = average_stats_df(4, team_list, team_fixture_id_dict, game_stats)

#2 game sliding average.
df_ml_1 = average_stats_df(1, team_list, team_fixture_id_dict, game_stats)

#2 game sliding average.
df_ml_6 = average_stats_df(6, team_list, team_fixture_id_dict, game_stats)

#2 game sliding average.
df_ml_7 = average_stats_df(7, team_list, team_fixture_id_dict, game_stats)

#2 game sliding average.
df_ml_8 = average_stats_df(8, team_list, team_fixture_id_dict, game_stats)

#2 game sliding average.
df_ml_9 = average_stats_df(9, team_list, team_fixture_id_dict, game_stats)
        
#creating and saving the ml dataframe with a 5 game sliding average.
df_for_ml_5_v2 = creating_ml_df(df_ml_5)
with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{df_5_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_5_v2, myFile)

#creating and saving the ml dataframe with a 10 game sliding average.
df_for_ml_10_v2 = creating_ml_df(df_ml_10)
with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{df_10_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_10_v2, myFile)

    #creating and saving the ml dataframe with a 3 game sliding average.
df_for_ml_3_v2 = creating_ml_df(df_ml_3)
with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{df_3_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_3_v2, myFile)#creating and saving the ml dataframe with a 3 game sliding average.

df_for_ml_2_v2 = creating_ml_df(df_ml_2)
with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{df_2_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_2_v2, myFile)

df_for_ml_4_v2 = creating_ml_df(df_ml_4)
with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{df_4_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_4_v2, myFile)

df_for_ml_1_v2 = creating_ml_df(df_ml_1)
with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{df_1_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_1_v2, myFile)

df_for_ml_6_v2 = creating_ml_df(df_ml_6)
with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{df_6_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_6_v2, myFile)

df_for_ml_7_v2 = creating_ml_df(df_ml_7)
with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{df_7_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_7_v2, myFile)

df_for_ml_8_v2 = creating_ml_df(df_ml_8)
with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{df_8_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_8_v2, myFile)

df_for_ml_9_v2 = creating_ml_df(df_ml_9)
with open(f'/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/{df_9_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_9_v2, myFile)


#for Power BI
df_for_ml_10_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi_v10.csv', index='False')
df_for_ml_5_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi_v5.csv', index='False')
df_for_ml_2_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi_v2.csv', index='False')
df_for_ml_3_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi_v3.csv', index='False')
df_for_ml_4_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi_v4.csv', index='False')
df_for_ml_1_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi_v1.csv', index='False')
df_for_ml_6_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi_v6.csv', index='False')
df_for_ml_7_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi_v7.csv', index='False')
df_for_ml_8_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi_v8.csv', index='False')
df_for_ml_9_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi_v9.csv', index='False')



print('Done')




# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
