<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

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
model.add(Dense(100, activation='softmax'))
model.trainable = True     # 가중치 동결

model.summary()


import time
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=3,
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

model.fit(x_train, y_train, epochs = 20, batch_size = 128, verbose=1, validation_split=0.25, callbacks=[es, mcp])
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
로스 :  4.876941680908203
acc :  0.0115
걸린시간 14.18 초 이만큼걸렸다. 

동결 전
로스 :  2.712418794631958
acc :  0.3146
걸린시간 106.0 초 이만큼걸렸다.
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

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
model.add(Dense(100, activation='softmax'))
model.trainable = True     # 가중치 동결

model.summary()


import time
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=3,
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

model.fit(x_train, y_train, epochs = 20, batch_size = 128, verbose=1, validation_split=0.25, callbacks=[es, mcp])
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
로스 :  4.876941680908203
acc :  0.0115
걸린시간 14.18 초 이만큼걸렸다. 

동결 전
로스 :  2.712418794631958
acc :  0.3146
걸린시간 106.0 초 이만큼걸렸다.
>>>>>>> cd855f8 (message)
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

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
model.add(Dense(100, activation='softmax'))
model.trainable = True     # 가중치 동결

model.summary()


import time
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=3,
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

model.fit(x_train, y_train, epochs = 20, batch_size = 128, verbose=1, validation_split=0.25, callbacks=[es, mcp])
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
로스 :  4.876941680908203
acc :  0.0115
걸린시간 14.18 초 이만큼걸렸다. 

동결 전
로스 :  2.712418794631958
acc :  0.3146
걸린시간 106.0 초 이만큼걸렸다.
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
"""