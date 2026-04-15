import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time

#1 데이터

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

y_test = y_test.to_numpy()

#2 모델 구성

model = Sequential()
model.add(Conv2D(64, (2,2), activation='relu', input_shape = (28,28,1,),
                 strides=1,
                 padding='same')) # (27,27,64)
model.add(MaxPooling2D())
model.add(Conv2D(filters = 64, activation='relu', kernel_size= (2,2),
                 strides=1,
                 padding='same')) #26,26,64
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu',
                 strides=1,
                 padding='same')) #25, 25, 32
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=16, input_shape=(32,), activation='relu')) 
model.add(Dense(10, activation='softmax'))

#3 컴파일 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=30,
    verbose=1,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras37/_fashion/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )


model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.25, callbacks=[es, mcp])

# 예측, 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss[0])
print('acc 스코어 :', round(acc, 3))
print(y_predict)

# 로스 :  0.26566705107688904
# acc 스코어 : 0.907

# Maxpooling
# 로스 :  0.2381344437599182
# acc 스코어 : 0.921