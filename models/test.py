import pandas as pd
import autokeras as ak
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from os import listdir
import time


df5_2022 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/2022_df_for_powerbi_v5.csv')

df5 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v5.csv')

id = df5_2022['Fixture ID']
print(id)

# x_test = df5.drop(df5[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator', 'Team Result Indicator']], axis=1)
# x_test = x_test.astype(int)

# y_test = df5.drop(df5[['Team Result Indicator']], axis=1)
# y_test = y_test.astype(int)

# x_test2 = df5_2022.drop(df5[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator', 'Team Result Indicator']], axis=1)
# x_test2 = x_test2.astype(int)

# y_test2 = df5.drop(df5[['Team Result Indicator']], axis=1)
# y_test2 = y_test2.astype(int)

# x5 = df5.drop(df5[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
# x5 = x5.astype(int)
# y5 = df5['Team Result Indicator'].astype(int)
# y5 = y5.astype(int)

# x_train5, x_test5, y_train5, y_test5 = train_test_split(x5, y5, test_size=0.2)


# model_0 = tf.keras.models.load_model('models/2022/ak_model_5/17')
# model_1 = tf.keras.models.load_model('models/2022/ak_model_5/14')
# model_2 = tf.keras.models.load_model('models/2022/ak_model_5/18')
# model_3 = tf.keras.models.load_model('models/2022/ak_model_5/19')
# model_4 = tf.keras.models.load_model('models/full/ak_model_5/4')
# model_5 = tf.keras.models.load_model('models/2022/ak_model_5/8')
# model_6 = tf.keras.models.load_model('models/2022/ak_model_5/7')
# model_7 = tf.keras.models.load_model('models/2022/ak_model_5/5')



# # y_pred = model_2.predict(x_test)
# # y_pred_2022 = y_pred.argmax(axis=-1)
# # acc_2022 = accuracy_score(y_test, y_pred_2022)
# # f1_2022 = f1_score(y_test, y_pred_2022, average='macro')
# # precision_2022 = precision_score(y_test, y_pred_2022, average='macro')
# # recall_2022 = recall_score(y_test, y_pred_2022, average='macro')

# # y_pred = model_4.predict(x_test)
# # y_pred = y_pred.argmax(axis=-1)
# # acc = accuracy_score(y_test, y_pred)
# # f1 = f1_score(y_test, y_pred, average='macro')
# # precision = precision_score(y_test, y_pred, average='macro')
# # recall = recall_score(y_test, y_pred, average='macro')


# # print(acc_2022, f1_2022, precision_2022, recall_2022)
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

