from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

"""
print(x)
print(y)
print(x.shape)  #(442, 10) (442,)
"""
#실습 맹글어라
#R2 0.62 이상

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=171)

#모델
model = Sequential()
model.add(Dense(1, input_dim=10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x_test, y_test, epochs=700, batch_size=10)

#4. 평가예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 :", r2)