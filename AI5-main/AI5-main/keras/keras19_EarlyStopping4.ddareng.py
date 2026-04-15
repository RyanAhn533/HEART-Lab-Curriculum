from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time as t
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping

path = "./_data/dacon/따릉이/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

train_csv = train_csv.dropna() #train_csv 데이터에서 결측치 삭제
test_csv = test_csv.fillna(test_csv.mean()) #test_csv에는 결측치 평균으로 넣기
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=3, train_size=0.7)
#x_train, x_val, y_train, y_val = train_test_split(x, y, shuffle=True, random_state=5, train_size=0.2)
#모델구성

model = Sequential()
model.add(Dense(16, input_dim=9))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patiecne=10, restore_best_weight=True)
model.compile(loss = 'mse', optimizer='adam')
st = t.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size = 16, verbose=1, 
                 validation_split=0.3, callbacks=[es])
edn = t.time()

#평가예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('로스값은 : ', loss)
print('y값은? ', y_predict)
print('r2값은?', r2)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('따릉이에옹')
"""
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
"""
plt.xlabel('epochs')
plt.xlabel('loss')
plt.grid()
plt.show()

print("#############걸린시간###########")
print('걸리는 시간!', round(edn-st, 1), "초")