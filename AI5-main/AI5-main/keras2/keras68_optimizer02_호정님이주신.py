from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.metrics import r2_score
import tensorflow as tf
import random as rn
rn.seed(337)    # 파이선의 랜덤 함수 seed 고정
tf.random.set_seed(337)
np.random.seed(337)     # 넘파이는 요롷게

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

# one hot encoding
y = pd.get_dummies(y)
# print(y.shape)  # (581012, 7)
# print(y)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75,
    random_state=337,
)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(16))
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

for i in range(6): 
    learning_rate = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # learning_rate = 0.0007       # default = 0.001
    learning_rate = learning_rate[i]
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1, verbose=0,
          batch_size=512,
          )

#4. 평가,예측
    print("=================1. 기본출력 ========================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 :{1}'.format(learning_rate, loss))

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))


# =================1. 기본출력 ========================
# lr : 0.1, 로스 :0.6625642776489258
# lr : 0.1, r2 : 0.3324335772409916
# =================1. 기본출력 ========================
# lr : 0.01, 로스 :0.6383678913116455
# lr : 0.01, r2 : 0.35466105300066875
# =================1. 기본출력 ========================
# lr : 0.005, 로스 :0.6304730176925659
# lr : 0.005, r2 : 0.36862279953461174
# =================1. 기본출력 ========================
# lr : 0.001, 로스 :0.6269198060035706
# lr : 0.001, r2 : 0.37802690905208103
# =================1. 기본출력 ========================
# lr : 0.0005, 로스 :0.6264317631721497
# lr : 0.0005, r2 : 0.37911791701326203
# =================1. 기본출력 ========================
# lr : 0.0001, 로스 :0.6262998580932617
# lr : 0.0001, r2 : 0.3793816695325363
