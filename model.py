import pandas as pd
import autokeras as ak
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

df = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v5.csv')

x = df.drop(df[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x = x.astype(int)
y = df['Team Result Indicator'].astype(int)
y = y.astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# model = ak.StructuredDataClassifier(max_trials=1)
# model.fit(x_train, y_train)
