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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1/255,

horizontal_flip=True, #수평 뒤집기
    vertical_flip=True, #수직뒤집기
    width_shift_range=0.1, #평행이동
    height_shift_range=0.1, #평행이동 수직
    rotation_range=5, #정해진 각도만큼 이미지 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, # 좌표
    fill_mode="nearest", #비율에 맞춰서 채워라
    
    )
train_datagen = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/image/horse_human/'


xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
class_mode='sparse',
color_mode='rgb',
shuffle=True)


x_train,x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#(718, 100, 100, 3) (309, 100, 100, 3) (718,) (309,)


# 원-핫 인코딩을 to_categorical로 적용
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)
print("원-핫 인코딩 후:", y_train.shape, y_test.shape)

from tensorflow.keras.applications import VGG16


vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(100, 100 ,3),
              )



model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))
model.trainable = True     # 가중치 동결


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

loss, accuracy = model.evaluate(x_test, y_test)
y_predict = (model.predict(x_test) > 0.5).astype(int)  #
acc = accuracy_score(y_test, y_predict)
print('로스 : ', loss)
print('acc : ', acc)
print("걸린시간", round(end-start,2), "초 이만큼걸렸다.")

"""

    
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1/255,

horizontal_flip=True, #수평 뒤집기
    vertical_flip=True, #수직뒤집기
    width_shift_range=0.1, #평행이동
    height_shift_range=0.1, #평행이동 수직
    rotation_range=5, #정해진 각도만큼 이미지 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, # 좌표
    fill_mode="nearest", #비율에 맞춰서 채워라
    
    )
train_datagen = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/image/horse_human/'


xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
class_mode='sparse',
color_mode='rgb',
shuffle=True)


x_train,x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#(718, 100, 100, 3) (309, 100, 100, 3) (718,) (309,)


# 원-핫 인코딩을 to_categorical로 적용
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)
print("원-핫 인코딩 후:", y_train.shape, y_test.shape)

from tensorflow.keras.applications import VGG16


vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(100, 100 ,3),
              )



model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))
model.trainable = True     # 가중치 동결


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

loss, accuracy = model.evaluate(x_test, y_test)
y_predict = (model.predict(x_test) > 0.5).astype(int)  #
acc = accuracy_score(y_test, y_predict)
print('로스 : ', loss)
print('acc : ', acc)
print("걸린시간", round(end-start,2), "초 이만큼걸렸다.")

"""

    
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1/255,

horizontal_flip=True, #수평 뒤집기
    vertical_flip=True, #수직뒤집기
    width_shift_range=0.1, #평행이동
    height_shift_range=0.1, #평행이동 수직
    rotation_range=5, #정해진 각도만큼 이미지 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, # 좌표
    fill_mode="nearest", #비율에 맞춰서 채워라
    
    )
train_datagen = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/image/horse_human/'


xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
class_mode='sparse',
color_mode='rgb',
shuffle=True)


x_train,x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#(718, 100, 100, 3) (309, 100, 100, 3) (718,) (309,)


# 원-핫 인코딩을 to_categorical로 적용
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)
print("원-핫 인코딩 후:", y_train.shape, y_test.shape)

from tensorflow.keras.applications import VGG16


vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(100, 100 ,3),
              )



model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))
model.trainable = True     # 가중치 동결


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

loss, accuracy = model.evaluate(x_test, y_test)
y_predict = (model.predict(x_test) > 0.5).astype(int)  #
acc = accuracy_score(y_test, y_predict)
print('로스 : ', loss)
print('acc : ', acc)
print("걸린시간", round(end-start,2), "초 이만큼걸렸다.")

"""

    
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
"""