import pandas as pd



fixture_2019= pd.read_csv('/home/michael/Desktop/ff/v2/prem_clean_fixtures_and_dataframes/2019_premier_league_fixtures_df.csv')
fixture_2020= pd.read_csv('/home/michael/Desktop/ff/v2/prem_clean_fixtures_and_dataframes/2020_premier_league_fixtures_df.csv')
fixture_2021= pd.read_csv('/home/michael/Desktop/ff/v2/prem_clean_fixtures_and_dataframes/2021_premier_league_fixtures_df.csv')
fixture_2022 = pd.read_csv('/home/michael/Desktop/ff/v2/prem_clean_fixtures_and_dataframes/2022_premier_league_fixtures_df.csv')

fixtures_clean_combined = pd.concat([fixture_2019, fixture_2020, fixture_2021, fixture_2022])
fixtures_clean_combined = fixtures_clean_combined.reset_index(drop=True)

fixtures_clean_combined.to_csv('/home/michael/Desktop/ff/v2/prem_clean_fixtures_and_dataframes/2019_2020_2021_2022_premier_league_fixtures_df.csv', index=False)