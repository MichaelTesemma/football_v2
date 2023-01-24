import pandas as pd
import autokeras as ak
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
import time

df = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v10.csv')

x = df.drop(df[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x = x.astype(int)
y = df['Team Result Indicator'].astype(int)
y = y.astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# start=time.time()

# model = ak.StructuredDataClassifier(max_trials=3, seed=42, overwrite=True)
# model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64)

# model = model.export_model()

# for i in range(20):
#     model = ak.StructuredDataClassifier(max_trials=2000, seed=42, overwrite=True)
#     model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64)

#     model = model.export_model()

#     print(model.summary())
#     model.save(f'v2/saved_model10/ak_model_v10{i}')
#     print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')

ak_model = ak.StructuredDataClassifier(seed=0)

# # fix it buy calling adapt() and some magic i found on https://github.com/keras-team/autokeras/issues/1746


for i in range(14):
    model = tf.keras.models.load_model(f"saved_model3/ak_model_v3{i}",
    custom_objects=ak.CUSTOM_OBJECTS)

    dataset, validation_data = ak_model._convert_to_dataset(
    x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=32)

    ak_model.tuner.adapt(model, dataset)
    
    model.save(f"ak_models_v3/ak_model_v3{i}", save_format="tf")
    print(f'model number {i} is saved')

for i in range(14):
    model = tf.keras.models.load_model(f"ak_models_v3/ak_model_v3{i}",custom_objects=ak.CUSTOM_OBJECTS)


    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax(axis=-1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    with open('ak_models_v3/stats.txt', 'a') as f:
        f.write('Model number: ' + str(i) + '\n' + '\n')
        f.write('Accuracy: ' + str(acc) + '\n')
        f.write('F1 Score: ' + str(f1) + '\n')
        f.write('Precision: ' + str(precision) + '\n')
        f.write('Recall: ' + str(recall) + '\n' + '\n')