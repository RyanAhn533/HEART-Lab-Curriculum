<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)
x_train = x_train/255.
x_test = x_test/255.

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10


vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(32, 32 ,3),
              )



model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
model.trainable = True     # 가중치 동결

model.summary()

'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 14,777,098
Non-trainable params: 0
'''

# vgg16.trainable = False 추가
'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
'''

########## [실습] 3가지 비교하기 ##########
# 1. 이전에 본인이 한 최상의 결과
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True 
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 위의 2, 3번은 time 체크 까지
import time
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

start = time.time()

model.fit(x_train, y_train, epochs = 100, batch_size = 128, verbose=1, validation_split=0.25, callbacks=[es, mcp])
end = time.time()
#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)
print('로스 : ', loss[0])
print('acc : ', acc)
print("걸린시간", round(end-start,2), "초 이만큼걸렸다.")

"""
로스 :  2.612356662750244
acc :  0.1078
걸린시간 58.43 초 이만큼걸렸다.
    
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)
x_train = x_train/255.
x_test = x_test/255.

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10


vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(32, 32 ,3),
              )



model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
model.trainable = True     # 가중치 동결

model.summary()

'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 14,777,098
Non-trainable params: 0
'''

# vgg16.trainable = False 추가
'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
'''

########## [실습] 3가지 비교하기 ##########
# 1. 이전에 본인이 한 최상의 결과
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True 
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 위의 2, 3번은 time 체크 까지
import time
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

start = time.time()

model.fit(x_train, y_train, epochs = 100, batch_size = 128, verbose=1, validation_split=0.25, callbacks=[es, mcp])
end = time.time()
#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)
print('로스 : ', loss[0])
print('acc : ', acc)
print("걸린시간", round(end-start,2), "초 이만큼걸렸다.")

"""
로스 :  2.612356662750244
acc :  0.1078
걸린시간 58.43 초 이만큼걸렸다.
    
>>>>>>> cd855f8 (message)
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)
x_train = x_train/255.
x_test = x_test/255.

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10


vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(32, 32 ,3),
              )



model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
model.trainable = True     # 가중치 동결

model.summary()

'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 14,777,098
Non-trainable params: 0
'''

# vgg16.trainable = False 추가
'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
'''

########## [실습] 3가지 비교하기 ##########
# 1. 이전에 본인이 한 최상의 결과
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True 
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 위의 2, 3번은 time 체크 까지
import time
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

start = time.time()

model.fit(x_train, y_train, epochs = 100, batch_size = 128, verbose=1, validation_split=0.25, callbacks=[es, mcp])
end = time.time()
#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)
print('로스 : ', loss[0])
print('acc : ', acc)
print("걸린시간", round(end-start,2), "초 이만큼걸렸다.")

"""
로스 :  2.612356662750244
acc :  0.1078
걸린시간 58.43 초 이만큼걸렸다.
    
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
"""