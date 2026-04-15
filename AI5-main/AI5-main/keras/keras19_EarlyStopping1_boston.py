from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_boston
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']



x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=3)

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=13))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = t.time()
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss', mode = 'min', #모르면 auto / min max auto
    patience=10, 
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=100, verbose = 2, 
                 batch_size=32, validation_split=0.3,
                 callbacks=[es]) # valtidation_data=(x_val, y_val)
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
exit()
import matplotlib.pyplot as plt
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

print('#######걸린시간############')
print('걸린시간은? ', round(end - start,1), '초')