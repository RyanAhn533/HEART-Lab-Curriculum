from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf
tf.random.set_seed(337)
np.random.seed(337)

datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer

scaler = StandardScaler()
x = scaler.fit_transform(x)


from sklearn.model_selection import train_test_split

x_train, x_test ,y_train, y_test = train_test_split(x, y, train_size=0.8,random_state=135)

model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.losses import mse, mae
for i in range(6): 
    lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # learning_rate = 0.0007       # default = 0.001
    learning_rate = lr[i]
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

# learning_rate = 0.001 #default
# learning_rate = 0.01

# model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=32,verbose=0)

#4 평가 예측
#4. 평가,예측
    print("=================1. 기본출력 ========================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 :{1}'.format(learning_rate, loss))

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))


# 로스 : {0} 23.388708114624023
# r2 : {0} 0.6760959209028634

# 로스 : {0} 23.388708114624023
# r2 : {0} 0.6760959209028634

# 로스 : {0} 23.07132339477539
# r2 : {0} 0.6804913184922805

# adagrad
# 로스 : {0} 24.20473861694336
# r2 : {0} 0.6647949456565236

# =================1. 기본출력 ========================
# lr : 0.005, 로스 :2.678703140190919e-06
# lr : 0.005, r2 : -7.146996744351764
# =================1. 기본출력 ========================
# lr : 0.001, 로스 :2.678703140190919e-06
# lr : 0.001, r2 : -7.146996744351764
# =================1. 기본출력 ========================
# lr : 0.0005, 로스 :2.678703140190919e-06
# lr : 0.0005, r2 : -7.146996744351764
# =================1. 기본출력 ========================
# lr : 0.0001, 로스 :2.678703140190919e-06
# lr : 0.0001, r2 : -7.146996744351764



