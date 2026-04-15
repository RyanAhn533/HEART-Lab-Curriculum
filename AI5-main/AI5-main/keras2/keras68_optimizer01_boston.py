from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf
import random as rn

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)
#모델
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.75, 
                                                    random_state=337)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0095
#learning late default 0.001
#learning late default 0.0001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
#Adam = 로스를 쳐줄여줌
model.fit(x_train, y_train,
          validation_split=0.2, epochs=100,
          batch_size=32)

#4.평가예측

loss = model.evaluate(x_test, y_test, verbose=0)


y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('lr : {0}, 로스 : {1}'.format(learning_rate, loss))
print('lr : {0}, r2 : {1}'.format(learning_rate, r2))

