import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=433, train_size=0.7)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, random_state=24, train_size=0.3)

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#컴파일, 훈련
#model.fit 에서 다른거 안하고 분리할 수 있음
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=762, verbose=0, validation_data=(x_val, y_val))

#결과
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(17)
print("예측값 : ", y_predict)
