import tensorflow as tf
import autokeras as ak
import pandas as pd
from sklearn.model_selection import train_test_split


# df5 = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v5.csv')

# x5 = df5.drop(df5[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
# x5 = x5.astype(int)
# y5 = df5['Team Result Indicator'].astype(int)
# y5 = y5.astype(int)

# x_train5, x_test5, y_train5, y_test5 = train_test_split(x5, y5, test_size=0.2)

# model = tf.keras.models.load_model('/home/michael/Desktop/v2/models2/2022/ak_model_5/17', custom_objects=ak.CUSTOM_OBJECTS)

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
# model.fit(x_train5, y_train5, validation_data=(x_test5, y_test5), batch_size=8, epochs=100000, callbacks=[early_stopping])

# model = tf.keras.models.load_model('/home/michael/Desktop/v2/models2/2022/ak_model_5/17', custom_objects=ak.CUSTOM_OBJECTS)
# print(model.summary())

model = tf.keras.models.load_model('/home/michael/Desktop/v2/models/test_model')
print(model.summary())