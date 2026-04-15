from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time as t
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from matplotlib import rc
#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
# 준형바보



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=9)

#2.모델
model = Sequential()
model.add(Dense(1, input_dim=8))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#컴파일, 훈련
es = EarlyStopping(monitor= 'val_loss', mode = 'min', 
                   patience=10, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
st = t.time()
hist = model.fit(x_train, y_train, epochs=40, batch_size=32, 
                 verbose=1, validation_split=0.3, callbacks=[es])
ed = t.time()

#4. 평가예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 :", r2)   #0.563
print("로스 :", loss)
print("로스 :",hist.history('val_loss'))


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))        #그림판의 사이즈를 9 by6으로
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('보스통 로스')
plt.xlabel('epochs')
plt.xlabel('loss')
plt.grid()
plt.show()

