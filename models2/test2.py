import pandas as pd
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score


df = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/2022_df_for_powerbi_v5.csv')

x = df.drop(df[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator', 'Team Result Indicator']], axis=1)
x = x.astype(int)

y = df['Team Result Indicator'].astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)

model = tf.keras.models.load_model('models/2022/ak_model_5/17', custom_objects=ak.CUSTOM_OBJECTS)

accuracy_list = []
for i in range(10):
    prediction = model.predict(x).argmax(axis=-1)
    accuracy = accuracy_score(y, prediction)
    accuracy_list.append(accuracy)


print(sum(accuracy_list)/len(accuracy_list))