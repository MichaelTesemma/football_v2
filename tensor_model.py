import pandas as pd
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from v2.model_v5 import x_train, y_train, x_test, y_test

dataset_10 = pd.read_csv('v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v10.csv')
dataset_5 = pd.read_csv('v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v5.csv')
dataset_3 = pd.read_csv('v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v3.csv')
dataset_2 = pd.read_csv('v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v2.csv')

x_2 = dataset_2.drop(dataset_2[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x_2 = x_2.astype(int)
y_2 = dataset_2['Team Result Indicator'].astype(int)
y_2 = y_2.astype(int)

x_3 = dataset_3.drop(dataset_3[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x_3 = x_3.astype(int)
y_3 = dataset_3['Team Result Indicator'].astype(int)
y_3 = y_3.astype(int)

x_5 = dataset_5.drop(dataset_5[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x_5 = x_5.astype(int)
y_5 = dataset_5['Team Result Indicator'].astype(int)
y_5 = y_5.astype(int)

x_10 = dataset_10.drop(dataset_10[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x_10 = x_10.astype(int)
y_10 = dataset_10['Team Result Indicator'].astype(int)
y_10 = y_10.astype(int)



model = tf.keras.models.load_model(f"v2/ak_models/ak_model12",custom_objects=ak.CUSTOM_OBJECTS)


y_pred = model.predict(x_2)
y_pred = y_pred.argmax(axis=-1)

acc = accuracy_score(y_2, y_pred)
f1 = f1_score(y_2, y_pred, average='macro')
precision = precision_score(y_2, y_pred, average='macro')
recall = recall_score(y_2, y_pred, average='macro')

print(f'2 Matches:  {acc}, {f1}, {precision}, {recall}')

y_pred = model.predict(x_3)
y_pred = y_pred.argmax(axis=-1)

acc = accuracy_score(y_3, y_pred)
f1 = f1_score(y_3, y_pred, average='macro')
precision = precision_score(y_3, y_pred, average='macro')
recall = recall_score(y_3, y_pred, average='macro')

print(f'3 Matches:  {acc}, {f1}, {precision}, {recall}')


y_pred = model.predict(x_5)
y_pred = y_pred.argmax(axis=-1)

acc = accuracy_score(y_5, y_pred)
f1 = f1_score(y_5, y_pred, average='macro')
precision = precision_score(y_5, y_pred, average='macro')
recall = recall_score(y_5, y_pred, average='macro')

print(f'5 Matches:  {acc}, {f1}, {precision}, {recall}')


y_pred = model.predict(x_10)
y_pred = y_pred.argmax(axis=-1)

acc = accuracy_score(y_10, y_pred)
f1 = f1_score(y_10, y_pred, average='macro')
precision = precision_score(y_10, y_pred, average='macro')
recall = recall_score(y_10, y_pred, average='macro')

print(f'10 Matches:  {acc}, {f1}, {precision}, {recall}')

# print(model.summary())

