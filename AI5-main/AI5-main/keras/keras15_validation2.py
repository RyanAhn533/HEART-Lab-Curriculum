import numpy as np

##잘라라!!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as np

x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
y = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
print(x)

x_train = x[:6]
y_train = y[:6]

print(x_train)
x_val = x[7:11]
y_val = y[7:11]

x_test = x[11:17]
y_test = y[11:17]

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1,
          verbose=1, validation_data=(x_val, y_val))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 10, batch_size=1)

loss = model.evaluate(x_test, y_test)
results = model.predict([18])
print("로스 : ", loss)
print("[11]의 예측값 : " ,results)