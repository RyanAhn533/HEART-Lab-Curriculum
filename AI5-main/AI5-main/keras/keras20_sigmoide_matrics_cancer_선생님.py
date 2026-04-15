import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import time as t
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_breast_cancer()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

#0과1의 갯수가 몇개인지 찾아요
print(np.unique(y, return_counts=True))
print(type(x))
print(pd.DataFrame(y).value_counts())
#print(y.value_counts) -> 에러
print(pd.Series(y))
#data 0개는 스칼라 0차원 1개는 백터 - 1차원, 2개이상은 행렬 - matrics 2차원  대괄호하나 더 늘어나는 후 부터 텐서, 3차원 텐서 / 이후 4차원 텐서 등등
#벡터형태를 시리즈라고함 / 2차원 행렬을 dataframe 이라고함
#pandas에 데이터 형태 백터 1나짜리. dataframe
print(pd.Series(y).value_counts)
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=3)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


#2. 모델
model = Sequential()
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) #accuracy, mse
#accuracy = 반올림
start = t.time()
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss', mode = 'min', #모르면 auto / min max auto
    patience=100, 
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=1000, batch_size=8, 
                 verbose=1, validation_split=0.2, callbacks=[es])
end = t.time()
#평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 :", loss)

y_predict = model.predict(x_test)
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)


r2 = r2_score(y_test, y_predict) 
print("r2스코어 :", r2)

print(y_pred)

print(y_predict[:5])
print("로스 : ", loss[0])
print("ACC : ", round(loss[1], 2))
print("acc_score :" , accuracy_score)
print("걸린시간 : ", round(end - start, 2), "초")