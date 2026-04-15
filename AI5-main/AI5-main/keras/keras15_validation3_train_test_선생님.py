import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=133, train_size=0.65)
print(x_train, y_train)
print(x_test, y_test)

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=1, validation_split=0.3 )

#complie에서 x_val, y_val 데이터를 앞서 정하면서 번거롭게하지말고
#mode.fit 데이터를 분리해서 모델을 돌리자
#validation_data=(x_val, y_val)
#validation_split=0.3  x_train에서 30퍼의 데이터를 쓰겠다.

#4. 평가예측
loss = model.evaluate(x_test, y_test, verbose=0)
#ss: 0.3265
#1/1 [==============================] - 0s 44ms/step - loss: 0.3265
results = model.predict([18])
print("예측값은 : ", results)
print("로스는 : ", loss)