from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import autokeras as ak
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN

df = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v2.csv')
x = df.drop(df[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
x = x.astype(int)
y = df['Team Result Indicator'].astype(int)
y = y.astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

def test(model, aug):
    df = pd.read_csv('/home/michael/Desktop/v2/prem_clean_fixtures_and_dataframes/df_for_powerbi_v2.csv')
    x = df.drop(df[['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator']], axis=1)
    x = x.astype(int)
    y = df['Team Result Indicator'].astype(int)
    y = y.astype(int)

    x, y = aug.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    model = model()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # y_pred = y_pred.argmax(axis=-1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print(f'{model}, {aug}:  {acc}, {f1}, {precision}, {recall}')



model = KNeighborsClassifier(n_neighbors=3)
ros = RandomOverSampler()
sm = SMOTE()
rus = RandomUnderSampler()
bos = BorderlineSMOTE()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# y_pred = y_pred.argmax(axis=-1)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f'normal:  {acc}, {f1}, {precision}, {recall}')
test(model, sm)
test(model, rus)
test(model, ros)
test(model, bos)


model2 = RandomForestClassifier()

model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
# y_pred = y_pred.argmax(axis=-1)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f'normal:  {acc}, {f1}, {precision}, {recall}')
test(model2, sm)
test(model2, rus)
test(model2, ros)
test(model2, bos)