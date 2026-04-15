from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time as t
import matplotlib.pyplot as plt
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

#실습 맹글어라
#R2 0.62 이상


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=171)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=21)

#모델
model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,activation='linear'))

#컴파일
model.compile(loss='mse', optimizer='adam')
st = t.time()
hist = model.fit(x_test, y_test, epochs=500, batch_size=32, verbose=0, validation_data=(x_val, y_val))
end = t.time()

#4. 평가예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 :", r2)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('케긁 바이크')
plt.xlabel('epochs')
plt.xlabel('loss')
plt.grid()
plt.show()
print("####################걸린시간#############")
print('걸리는 시간은?', round(end-st, 1), "초")