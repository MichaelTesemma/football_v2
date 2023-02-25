import pandas as pd
import autokeras as ak
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from os import listdir
import time
# df1 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v1.csv')
df1 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v1.csv')
df2 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/2022_df_for_powerbi_v2.csv')
df3 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/2022_df_for_powerbi_v3.csv')
df4 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v4.csv')
df5 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v5.csv')
df6 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v6.csv')
df7 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/2022_df_for_powerbi_v7.csv')
df8 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v8.csv')
df9 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v9.csv')
df10 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/2022_df_for_powerbi_v10.csv')


x1 = df1.drop(df1[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x1 = x1.astype(int)
y1 = df1['Team Result Indicator'].astype(int)
y1 = y1.astype(int)

x2 = df2.drop(df2[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x2 = x2.astype(int)
y2 = df2['Team Result Indicator'].astype(int)
y2 = y2.astype(int)

x3 = df3.drop(df3[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x3 = x3.astype(int)
y3 = df3['Team Result Indicator'].astype(int)
y3 = y3.astype(int)

x4 = df4.drop(df4[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x4 = x4.astype(int)
y4 = df4['Team Result Indicator'].astype(int)
y4 = y4.astype(int)

x5 = df5.drop(df5[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x5 = x5.astype(int)
y5 = df5['Team Result Indicator'].astype(int)
y5 = y5.astype(int)

x6 = df6.drop(df6[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x6 = x6.astype(int)
y6 = df6['Team Result Indicator'].astype(int)
y6 = y6.astype(int)

x7 = df7.drop(df7[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x7 = x7.astype(int)
y7 = df7['Team Result Indicator'].astype(int)
y7 = y7.astype(int)

x8 = df8.drop(df8[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x8 = x8.astype(int)
y8 = df8['Team Result Indicator'].astype(int)
y8 = y8.astype(int)

x9 = df9.drop(df9[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x9 = x9.astype(int)
y9 = df9['Team Result Indicator'].astype(int)
y9 = y9.astype(int)

x10 = df10.drop(df10[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x10 = x10.astype(int)
y10 = df10['Team Result Indicator'].astype(int)
y10 = y10.astype(int)



x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.2)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.2)
x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.2)
x_train4, x_test4, y_train4, y_test4 = train_test_split(x4, y4, test_size=0.2)
x_train5, x_test5, y_train5, y_test5 = train_test_split(x5, y5, test_size=0.2)
x_train6, x_test6, y_train6, y_test6 = train_test_split(x6, y6, test_size=0.2)
x_train7, x_test7, y_train7, y_test7 = train_test_split(x7, y7, test_size=0.2)
x_train8, x_test8, y_train8, y_test8 = train_test_split(x8, y8, test_size=0.2)
x_train9, x_test9, y_train9, y_test9 = train_test_split(x9, y9, test_size=0.2)
x_train10, x_test10, y_train10, y_test10 = train_test_split(x10, y10, test_size=0.3)


start=time.time()




def auto_ml(folder1, folder2, folder3, x_train, y_train, x_test, y_test):
    x_train = x_train
    y_train = y_train
    x_test = x_test
    y_test = y_test
    # for i in range(5):
    #     model = ak.StructuredDataClassifier(max_trials=80, seed=42, overwrite=True)
    #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250, restore_best_weights=True)
    #     model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=100000, callbacks=[early_stopping])

    #     model = model.export_model()

    #     print(model.summary())
    #     model.save(f'{folder1}/{i}')

    # ak_model = ak.StructuredDataClassifier(seed=0)
    model = ak.StructuredDataClassifier(max_trials=400, seed=42, overwrite=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=100000, callbacks=[early_stopping])

    model = model.export_model()

    print(model.summary())
    model.save(f'{folder1}/0')

    ak_model = ak.StructuredDataClassifier(seed=0)

    # fixed it buy calling adapt() and some magic i found on https://github.com/keras-team/autokeras/issues/1746


    # for i in range(len(listdir(folder1))):
    #     model = tf.keras.models.load_model(f"{folder1}/{i}",
    #     custom_objects=ak.CUSTOM_OBJECTS)

    #     dataset, validation_data = ak_model._convert_to_dataset(
    #     x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=32)

    #     ak_model.tuner.adapt(model, dataset)
        
    #     model.save(f"{folder2}/{i}", save_format="tf")
    #     print(f'model number {i} is saved')
    model = tf.keras.models.load_model(f"{folder1}/0",
    custom_objects=ak.CUSTOM_OBJECTS)

    dataset, validation_data = ak_model._convert_to_dataset(
    x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=32)

    ak_model.tuner.adapt(model, dataset)
        
    model.save(f"{folder2}/0", save_format="tf")
    print(f'model number 0 is saved')

    # for i in range(len(listdir(folder2))):
    #     model = tf.keras.models.load_model(f"{folder2}/{i}",custom_objects=ak.CUSTOM_OBJECTS)


    #     y_pred = model.predict(x_test)
    #     y_pred = y_pred.argmax(axis=-1)

    #     acc = accuracy_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred, average='macro')
    #     precision = precision_score(y_test, y_pred, average='macro')
    #     recall = recall_score(y_test, y_pred, average='macro')

    #     with open(f'{folder3}/stats.txt', 'a') as f:
    #         f.write('Model number: ' + str(i) + '\n' + '\n')
    #         f.write('Accuracy: ' + str(acc) + '\n')
    #         f.write('F1 Score: ' + str(f1) + '\n')
    #         f.write('Precision: ' + str(precision) + '\n')
    #         f.write('Recall: ' + str(recall) + '\n' + '\n')
    model = tf.keras.models.load_model(f"{folder2}/0",custom_objects=ak.CUSTOM_OBJECTS)


    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax(axis=-1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    with open(f'{folder3}/stats.txt', 'a') as f:
        f.write('Model number: ' + str(50) + '\n' + '\n')
        f.write('Accuracy: ' + str(acc) + '\n')
        f.write('F1 Score: ' + str(f1) + '\n')
        f.write('Precision: ' + str(precision) + '\n')
        f.write('Recall: ' + str(recall) + '\n' + '\n')

auto_ml('models/full/saved_model_9', 'models/full/ak_model_9', 'models/full/model_9_stats', x_train9, y_train9, x_test9, y_test9)
auto_ml('models/full/saved_model_1', 'models/full/ak_model_1', 'models/full/model_1_stats', x_train1, y_train1, x_test1, y_test1)
auto_ml('models/full/saved_model_5', 'models/full/ak_model_5', 'models/full/model_5_stats', x_train5, y_train5, x_test5, y_test5)
auto_ml('models/full/saved_model_6', 'models/full/ak_model_6', 'models/full/model_6_stats', x_train6, y_train6, x_test6, y_test6)
auto_ml('models/full/saved_model_8', 'models/full/ak_model_8', 'models/full/model_8_stats', x_train8, y_train8, x_test8, y_test8)



print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')

