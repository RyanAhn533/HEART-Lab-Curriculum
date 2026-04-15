from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping
import time as t
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

path = "./_data/kaggle/Bank/"

train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0 )

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.7, random_state=41)
#2.모델 구성
model = Sequential()
model.add(Dense(30, input_dim=10, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=10,
                   restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 verbose=3, validation_split=0.1,
                 callbacks=[es])

#평가 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss 는 뭐게용?', loss[0]) #compile할때 metrics 추가했으니, loss값 2개
print('acc :', round(loss[1],3))
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2값이?', r2_score)
y_submit = model.predict(test_csv)
y_pred = np.round(y_submit)
