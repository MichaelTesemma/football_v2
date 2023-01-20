import glob
import pandas as pd

# Create an empty list to store the data frames
df_list = []

# Use glob to get a list of file paths for all CSV files in the current directory
csv_files = glob.glob('v2/data/Bundesliga/*.csv')

# Iterate over the file paths
for file in csv_files:
    # Read the CSV file into a data frame
    df = pd.read_csv(file)
    df = df.dropna(axis='index', how='all')
    
    # Split the scores into separate columns
    df[['Team1', 'Team2']] = df['Score'].str.split('–', expand=True)
    
    # Convert the scores to integers
    # df[['Team1', 'Team2']] = df[['Team1', 'Team2']].astype(int)
    
    # Add the results column
    df['Result'] = df.apply(lambda row: 1 if row['Team1'] > row['Team2'] else (0 if row['Team1'] == row['Team2'] else 2), axis=1)
    
    # Append the data frame to the list
    df_list.append(df)

# Concatenate all the data frames into a single data frame
df = pd.concat(df_list)
df.to_csv('v2/data/Bundesliga/all_bundesliga.csv')

print(df)

# Create an empty list to store the data frames
df_list = []

# Use glob to get a list of file paths for all CSV files in the current directory
csv_files = glob.glob('v2/data/Champions_League/*.csv')

# Iterate over the file paths
for file in csv_files:
    # Read the CSV file into a data frame
    df = pd.read_csv(file)
    df = df.dropna(axis='index', how='all')
    
    # Split the scores into separate columns
    df[['Team1', 'Team2']] = df['Score'].str.split('–', expand=True)
    
    # Convert the scores to integers
    # df[['Team1', 'Team2']] = df[['Team1', 'Team2']].astype(int)
    
    # Add the results column
    df['Result'] = df.apply(lambda row: 1 if row['Team1'] > row['Team2'] else (0 if row['Team1'] == row['Team2'] else 2), axis=1)
    
    # Append the data frame to the list
    df_list.append(df)

# Concatenate all the data frames into a single data frame
df = pd.concat(df_list)
df.to_csv('v2/data/Bundesliga/all_champions_league.csv')

print(df)

# Create an empty list to store the data frames
df_list = []

# Use glob to get a list of file paths for all CSV files in the current directory
csv_files = glob.glob('v2/data/Europa_League/*.csv')

# Iterate over the file paths
for file in csv_files:
    # Read the CSV file into a data frame
    df = pd.read_csv(file)
    df = df.dropna(axis='index', how='all')
    
    # Split the scores into separate columns
    df[['Team1', 'Team2']] = df['Score'].str.split('–', expand=True)
    
    # Convert the scores to integers
    # df[['Team1', 'Team2']] = df[['Team1', 'Team2']].astype(int)
    
    # Add the results column
    df['Result'] = df.apply(lambda row: 1 if row['Team1'] > row['Team2'] else (0 if row['Team1'] == row['Team2'] else 2), axis=1)
    
    # Append the data frame to the list
    df_list.append(df)

# Concatenate all the data frames into a single data frame
df = pd.concat(df_list)
df.to_csv('v2/data/Bundesliga/all_europa_league.csv')

print(df)

# Create an empty list to store the data frames
df_list = []

# Use glob to get a list of file paths for all CSV files in the current directory
csv_files = glob.glob('v2/data/La_Liga/*.csv')

# Iterate over the file paths
for file in csv_files:
    # Read the CSV file into a data frame
    df = pd.read_csv(file)
    df = df.dropna(axis='index', how='all')
    
    # Split the scores into separate columns
    df[['Team1', 'Team2']] = df['Score'].str.split('–', expand=True)
    
    # Convert the scores to integers
    # df[['Team1', 'Team2']] = df[['Team1', 'Team2']].astype(int)
    
    # Add the results column
    df['Result'] = df.apply(lambda row: 1 if row['Team1'] > row['Team2'] else (0 if row['Team1'] == row['Team2'] else 2), axis=1)
    
    # Append the data frame to the list
    df_list.append(df)

# Concatenate all the data frames into a single data frame
df = pd.concat(df_list)
df.to_csv('v2/data/Bundesliga/all_la_liga.csv')

print(df)

# Create an empty list to store the data frames
df_list = []

# Use glob to get a list of file paths for all CSV files in the current directory
csv_files = glob.glob('v2/data/Premier_League/*.csv')

# Iterate over the file paths
for file in csv_files:
    # Read the CSV file into a data frame
    df = pd.read_csv(file)
    df = df.dropna(axis='index', how='all')
    
    # Split the scores into separate columns
    df[['Team1', 'Team2']] = df['Score'].str.split('–', expand=True)
    
    # Convert the scores to integers
    # df[['Team1', 'Team2']] = df[['Team1', 'Team2']].astype(int)
    
    # Add the results column
    df['Result'] = df.apply(lambda row: 1 if row['Team1'] > row['Team2'] else (0 if row['Team1'] == row['Team2'] else 2), axis=1)
    
    # Append the data frame to the list
    df_list.append(df)

# Concatenate all the data frames into a single data frame
df = pd.concat(df_list)
df.to_csv('v2/data/Bundesliga/all_premier_league.csv')

print(df)

# Create an empty list to store the data frames
df_list = []

# Use glob to get a list of file paths for all CSV files in the current directory
csv_files = glob.glob('v2/data/Serie_A/*.csv')

# Iterate over the file paths
for file in csv_files:
    # Read the CSV file into a data frame
    df = pd.read_csv(file)
    df = df.dropna(axis='index', how='all')
    
    # Split the scores into separate columns
    df[['Team1', 'Team2']] = df['Score'].str.split('–', expand=True)
    
    # Convert the scores to integers
    # df[['Team1', 'Team2']] = df[['Team1', 'Team2']].astype(int)
    
    # Add the results column
    df['Result'] = df.apply(lambda row: 1 if row['Team1'] > row['Team2'] else (0 if row['Team1'] == row['Team2'] else 2), axis=1)
    
    # Append the data frame to the list
    df_list.append(df)

# Concatenate all the data frames into a single data frame
df = pd.concat(df_list)
df.to_csv('v2/data/Bundesliga/all_serie_a.csv')

print(df)

df_bundesliga = pd.read_csv('v2/data/Bundesliga/all_bundesliga.csv', low_memory=False)
df_champions_league = pd.read_csv('v2/data/Bundesliga/all_champions_league.csv')
df_europa_league = pd.read_csv('v2/data/Bundesliga/all_europa_league.csv')
df_la_liga = pd.read_csv('v2/data/Bundesliga/all_la_liga.csv')
df_premier_league = pd.read_csv('v2/data/Bundesliga/all_premier_league.csv')
df_serie_a = pd.read_csv('v2/data/Bundesliga/all_serie_a.csv')

df_list = [df_bundesliga, df_champions_league, df_europa_league, df_la_liga, df_premier_league, df_serie_a]

df = pd.concat(df_list)

df.to_csv('v2/data/all_combined.csv')
