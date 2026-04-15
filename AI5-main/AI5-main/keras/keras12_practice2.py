import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
"""
path = "./_data/따릉이/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.95, shuffle=True, random_state=132)
"""
train_csv = pd.read_csv("./_data/따릉이/train.csv", index_col=0)
test_csv = pd.read_csv("./_data/따릉이/test.csv", index_col=0)
submission_csv = pd.read_csv("./_data/따릉이/submission.csv", index_col=0)
train_csv = train_csv.dropna()
test_csv + test.csv.fillna(test_csv.mean())
x = train_csv.drop(['count'], axis=1)
y = train_csv(['count'])
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=174)

loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_0716_1728.csv")