import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1.데이터
datasets = load_iris()
#print(datasets)    
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)
print(y) # 순서대로니까 shuiffle해야함
"""
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))
"""
# 0    50
# 1    50
# 2    50
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.transform(y)

x = pd.concat([pd.DataFrame(x)])
y = pd.DataFrame(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, shuffle=True)

model = Sequential()
model.add(Dense(16, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

#컴파일, 훈련
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=100, restore_best_weights=True)
#평가예측
model.compile(loss = 'mse', optimizer= 'adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10)

loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 는 뭐게?", r2)