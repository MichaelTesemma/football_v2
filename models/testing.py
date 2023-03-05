import pandas as pd
import autokeras as ak
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from os import listdir

df10 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v10.csv')

x10 = df10.drop(df10[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x10 = x10.astype(int)
y10 = df10['Team Result Indicator'].astype(int)
y10 = y10.astype(int)

x_train10, x_test10, y_train10, y_test10 = train_test_split(x10, y10, test_size=0.2)


dir = listdir('/home/michael/Desktop/v2/models/models/2022/ak_model_10')
print(dir)

model = tf.keras.models.load_model(f"/home/michael/Desktop/v2/models/models/2022/ak_model_10/0",custom_objects=ak.CUSTOM_OBJECTS)

ak_model = ak.StructuredDataClassifier(seed=0)
dataset, validation_data = ak_model._convert_to_dataset(x=x_train10, y=y_train10, validation_data=(x_test10, y_test10), batch_size=102)
ak_model.tuner.adapt(model, dataset)

# y_pred = model.predict(x_test10)
# y_pred = y_pred.argmax(axis=-1)

# acc  = accuracy_score(y_test10, y_pred)
# f1 = f1_score(y_test10, y_pred, average='macro')
# precision = precision_score(y_test10, y_pred, average='macro')
# recall = recall_score(y_test10, y_pred, average='macro')
# print(acc, f1, precision, recall)

def compile(folder2, x_test, y_test, folder10):
    for i in range(len(listdir(folder2))):
        model = tf.keras.models.load_model(f"{folder2}/{i}",custom_objects=ak.CUSTOM_OBJECTS)
       
        ak_model = ak.StructuredDataClassifier(seed=0)
        dataset, validation_data = ak_model._convert_to_dataset(x=x_train10, y=y_train10, validation_data=(x_test10, y_test10), batch_size=102)
        ak_model.tuner.adapt(model, dataset)

        y_pred = model.predict(x_test)
        y_pred = y_pred.argmax(axis=-1)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        with open(f'{folder10}/stats.txt', 'a') as f:
            f.write('Model number: ' + str(i) + '\n' + '\n')
            f.write('Accuracy: ' + str(acc) + '\n')
            f.write('F1 Score: ' + str(f1) + '\n')
            f.write('Precision: ' + str(precision) + '\n')
            f.write('Recall: ' + str(recall) + '\n' + '\n')

compile('models/2022/ak_model_10', x_test10, y_test10, 'models/2022/ak_model_10')