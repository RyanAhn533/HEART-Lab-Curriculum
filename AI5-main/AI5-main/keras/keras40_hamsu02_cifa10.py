#accuracy 95프로 만들기
#cifar10 뭔지 찾아보기

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 


# (50000, 32, 32, 3)
# (10000, 32, 32, 3)
# (50000, 1)
# (10000, 1)

x_train = x_train/255.
x_test = x_test/255.
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# x_train = x_train.reshape(50000, 32, 23, 3)
# x_test = x_test.reshape()

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# y_train = y_train.to_numpy()
# y_test = y_test.to_numpy()


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)


# import matplotlib.pyplot as plt
# plt.imshow(x_train[49999]) #xtrain의 59999번째를 보여주겠다 , 'gray' 색상을 흑백으로 하겠다. 
# plt.show()
# print('y_train[49999]의 값 : ', y_train[49999])

#2. 모델 구성
input1 = Input(shape=(32,32,3))
dense1 = Conv2D(64, (2,2), padding ='same', activation='relu')(input1)
dense2 = Conv2D(64, (3,3), padding='same', activation='relu')(dense1)
dense3 = Conv2D(32, (2,2), padding='same', activation='relu')(dense2)
dense4 = Conv2D(32, (2,2), padding='same', activation='relu')(dense3)
drop1 = Dropout(0.2)(dense4)
dense5 = Conv2D(32, (2,2), padding='same', activation='relu')(drop1)
maxp1 = MaxPooling2D()(dense5)
dense6 = Conv2D(32, (2,2), padding='same', activation='relu')(maxp1)
maxp2 = MaxPooling2D()(dense6)
Flat1 = Flatten()(maxp2)
dense7 = Dense(32, activation='relu')(Flat1)
drop2 = Dropout(0.2)(dense7)
dense8 = Dense(16, activation='relu')(drop2)
output1 = Dense(10, activation='softmax')(dense8)
model = Model(inputs = input1, outputs = output1)

#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=20,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras36/_cifa10/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor= 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only= True,
    filepath = filepath
)


model.fit(x_train, y_train, epochs = 100, batch_size = 128, verbose=1, validation_split=0.25, callbacks=[es, mcp])

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss[0])
print('acc : ', acc)

# padding
# 로스 :  1.2356504201889038
# acc :  0.558