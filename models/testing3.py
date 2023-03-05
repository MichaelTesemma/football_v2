import tensorflow as tf
import autokeras as ak
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score


df = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/2022_df_for_powerbi_v5.csv')

x = df.drop(df[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)

y = df['Team Result Indicator']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

y_train_cat = tf.keras.utils.to_categorical(y_train, 3)
y_test_cat = tf.keras.utils.to_categorical(y_test, 3)

og_model = tf.keras.models.load_model('/home/michael/Desktop/v2/models2/2022/ak_model_3/0', custom_objects=ak.CUSTOM_OBJECTS)
input, category, normalization, batch, drop, classification = og_model.layers[0], og_model.layers[1], og_model.layers[2], og_model.layers[4], og_model.layers[6], og_model.layers[8],
model = tf.keras.models.Sequential([
    input,
    category,
    normalization,
    tf.keras.layers.Dense(128),
    batch,
    tf.keras.layers.ReLU(128),
    drop,
    tf.keras.layers.Dense(3),
    classification
])


# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])
# model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat),epochs=100000, batch_size=4, callbacks=[early_stopping])
# model.save('/home/michael/Desktop/v2/models/2022_3')


model = tf.keras.models.load_model('/home/michael/Desktop/v2/models/test_model3')
y_pred = model.predict(x_test)
y_pred = y_pred.argmax(axis=-1)

acc  = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print(acc, f1, precision, recall)

# print(og_model.summary())
# print(model.summary())
