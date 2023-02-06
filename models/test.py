import pandas as pd
import autokeras as ak
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from os import listdir
import time


df5 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/2022_df_for_powerbi_v5.csv')


x_test = df5.drop(df5[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator', 'Team Result Indicator']], axis=1)
x_test = x_test.astype(int)

y_test = df5['Team Result Indicator']
y_test = y_test.astype(int)

x_train5, x_test5, y_train5, y_test5 = train_test_split(x_test, y_test, test_size=0.2)

df7 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/2022_df_for_powerbi_v7.csv')


x_test = df7.drop(df7[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator', 'Team Result Indicator']], axis=1)
x_test = x_test.astype(int)

y_test = df7['Team Result Indicator']
y_test = y_test.astype(int)

x_train7, x_test7, y_train7, y_test7 = train_test_split(x_test, y_test, test_size=0.2)

df5 = pd.read_csv('models/data/2022_df_for_powerbi_v5.csv')


x_test = df5.drop(df5[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator', 'Team Result Indicator']], axis=1)
x_test = x_test.astype(int)

y_test = df5['Team Result Indicator']
y_test = y_test.astype(int)

x_train5, x_test5, y_train5, y_test5 = train_test_split(x_test, y_test, test_size=0.2)


model_2022_2 = tf.keras.models.load_model('models/2022/ak_model_2/0')
model_2022_3 = tf.keras.models.load_model('models/2022/ak_model_3/0')
model_2022_4 = tf.keras.models.load_model('models/2022/ak_model_4/0')
model_2022_5_8 = tf.keras.models.load_model('models/2022/ak_model_5/8')
model_2022_5_17 = tf.keras.models.load_model('models/2022/ak_model_5/17')
model_2022_6 = tf.keras.models.load_model('models/2022/ak_model_6/0')
model_2022_7 = tf.keras.models.load_model('models/2022/ak_model_7/0')
model_2022_8 = tf.keras.models.load_model('models/2022/ak_model_8/0')
model_2022_9 = tf.keras.models.load_model('models/2022/ak_model_9/0')
model_2022_10 = tf.keras.models.load_model('models/2022/ak_model_10/0')

model_full_1 = tf.keras.models.load_model('models/full/ak_model_1/5') # under 60
model_full_2 = tf.keras.models.load_model('models/full/ak_model_2/9') # under 60
model_full_3 = tf.keras.models.load_model('models/full/ak_model_3/2') # under 60
model_full_4 = tf.keras.models.load_model('models/full/ak_model_4/0') # under 60
model_full_5 = tf.keras.models.load_model('models/full/ak_model_5/4') # under 60
model_full_6 = tf.keras.models.load_model('models/full/ak_model_6/0') # under 60
model_full_7 = tf.keras.models.load_model('models/full/ak_model_7/2')
model_full_8 = tf.keras.models.load_model('models/full/ak_model_8/0')
model_full_9 = tf.keras.models.load_model('models/full/ak_model_9/0')
model_full_10 = tf.keras.models.load_model('models/full/ak_model_10/0')



y_pred = model_2022_5_17.predict(x_test5)
y_pred = y_pred.argmax(axis=-1)
acc_2022 = accuracy_score(y_test5, y_pred)
f1_2022 = f1_score(y_test5, y_pred, average='macro')
precision_2022 = precision_score(y_test5, y_pred, average='macro')
recall_2022 = recall_score(y_test5, y_pred, average='macro')

# # y_pred = model_4.predict(x_test)
# # y_pred = y_pred.argmax(axis=-1)
# # acc = accuracy_score(y_test, y_pred)
# # f1 = f1_score(y_test, y_pred, average='macro')
# # precision = precision_score(y_test, y_pred, average='macro')
# # recall = recall_score(y_test, y_pred, average='macro')


print(acc_2022, f1_2022, precision_2022, recall_2022)
# # print(acc, f1, precision, recall)

# # print(y_pred_2022, len(y_pred_2022))
# # print(y_pred, len(y_pred))

# df = pd.DataFrame()


# y_pred = model_2.predict(x_test)
# y_pred = y_pred.argmax(axis=-1)

# y_pred2 = model_4.predict(x_test2)
# y_pred2 = y_pred2.argmax(axis=-1)

# df['model_5_2022'] = y_pred2
# df['model_5_full'] = y_pred
# df['full_result'] = df5['Team Result Indicator']
# df['2022_result'] = df5_2022['Team Result Indicator']

# print(df.head())

# # print(model_2.summary())

