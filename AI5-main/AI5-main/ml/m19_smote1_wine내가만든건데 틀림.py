import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(178,13), (178)

print(np.unique(y, return_counts=True))
print(pd.value_counts(y))
print(y)
x = x[:-40]
y = y[:-40]
print(np.unique(y, return_counts=True)) 
#(array([0, 1, 2]), array([59, 71,  8], dtype=int64))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, shuffle=True, random_state=333)


#모델
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#sparse_categirical_crossentropy 하면 원핫 안해도 가능함
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

#f1_score
y_predict = model.predict(x_test)
y_predict_classes = np.argmax(y_predict, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_classes, y_predict_classes)
f1 = f1_score(y_test_classes, y_predict_classes, average='macro')

print('acc : ', acc)
print('f1 : ', f1_score)
