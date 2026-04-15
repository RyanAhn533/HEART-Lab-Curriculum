from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_boston
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping


dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']



x = dataset.data
y = dataset.target

#train, test 후에 sclaer하면 test와 비교할 때 test는 scaler가 안 되어있기때문에
#scaler안한 test데이터와 train데이터 값의 차이가 생김??? 그래서 scaler 여기다가 해야하나?
#아니다~!

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=3)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델
model = Sequential()
#model.add(Dense(1, input_dim=13))
model.add(Dense(32, input_shape=(13,))) # 이미지 input_shape=(8,8,1)
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = t.time()
hist = model.fit(x_train, y_train, epochs=500, verbose = 2, batch_size=32, validation_split=0.3) # valtidation_data=(x_val, y_val)
end  = t.time()
#평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 :", loss)

y_predict = model.predict(x_test)


r2 = r2_score(y_test, y_predict) 
print("r2스코어 :", r2)


print("걸린시간은? ", round(end - start, 2), "초")
print("+++++++++++++++++++++++hist=====================")
print(hist)
print("===================hist.history===================")
print(hist.history)
print("================loss=============")
print(hist.history['loss'])
print("=============val_loss=================")
print(hist.history['val_loss'])

#딕셔너리란 key 와 value 가 존재하는 데이터 형이다.
# hist.history - loss : [] + val_loss : [] 이렇게 존재하는 것 처럼