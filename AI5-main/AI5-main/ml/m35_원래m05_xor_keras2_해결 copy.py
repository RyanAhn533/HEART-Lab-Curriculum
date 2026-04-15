import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0])
print(x_data.shape, y_data.shape)   # (4, 2) (4,)

#2. 모델
# model = LinearSVC()
# model = Perceptron()

model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=500)

#4. 평가, 예측
# acc = model.score(x_data, y_data)
# print('model.score :', acc)

loss = model.evaluate(x_data, y_data)
print('acc :', loss[1])

y_predict = model.predict(x_data)
# acc2 = accuracy_score(y_data, y_predict)    # ValueError: Classification metrics can't handle a mix of binary and continuous targets
y_predict = np.round(y_predict).reshape(-1,).astype(int)
acc2 = accuracy_score(y_data, y_predict)
print('accuracy_score :', acc2)

print('====================')
print(y_data)
print(y_predict)

'''
ValueError: Classification metrics can't handle a mix of binary and continuous targets
y_predict = np.round(y_predict).redhape(-1,).astype(int) 추가

acc : 0.75
accuracy_score : 0.75
====================
[0 1 1 0]
[0 1 1 1]

[실습] acc 1.0 만들기 -> relu 추가

acc : 1.0
accuracy_score : 1.0
====================
[0 1 1 0]
[0 1 1 0]

'''
