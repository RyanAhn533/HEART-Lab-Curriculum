from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
model.add(Dense(1, input_dim=10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x_test, y_test, epochs=100, batch_size=10, verbose=0, validation_data=(x_val, y_val))

#4. 평가예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 :", r2)