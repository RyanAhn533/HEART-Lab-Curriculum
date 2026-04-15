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
import time

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

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


start = time.time()

np_path = 'C:\\프로그램\\ai5\\_data\\_save\\save_npy\\'

x_train = np.load(np_path + "keras43_01_x_train1.npy")
y_train = np.load(np_path + "keras43_01_y_train1.npy")
x_test = np.load(np_path + "keras43_01_x_test.npy")
y_test = np.load(np_path + "keras43_01_y_test.npy")
x_train = x_train.reshape(25000,100*100,3)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=1186)
print(x_train)
print(x_train.shape)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10


vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(100, 100 ,3),
              )



model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
model.trainable = True     # 가중치 동결

model.summary()

import time
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
import time

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

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


start = time.time()

np_path = 'C:\\프로그램\\ai5\\_data\\_save\\save_npy\\'

x_train = np.load(np_path + "keras43_01_x_train1.npy")
y_train = np.load(np_path + "keras43_01_y_train1.npy")
x_test = np.load(np_path + "keras43_01_x_test.npy")
y_test = np.load(np_path + "keras43_01_y_test.npy")
x_train = x_train.reshape(25000,100*100,3)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=1186)
print(x_train)
print(x_train.shape)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10


vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(100, 100 ,3),
              )



model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
model.trainable = True     # 가중치 동결

model.summary()

import time
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
import time

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

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


start = time.time()

np_path = 'C:\\프로그램\\ai5\\_data\\_save\\save_npy\\'

x_train = np.load(np_path + "keras43_01_x_train1.npy")
y_train = np.load(np_path + "keras43_01_y_train1.npy")
x_test = np.load(np_path + "keras43_01_x_test.npy")
y_test = np.load(np_path + "keras43_01_y_test.npy")
x_train = x_train.reshape(25000,100*100,3)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=1186)
print(x_train)
print(x_train.shape)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10


vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(100, 100 ,3),
              )



model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
model.trainable = True     # 가중치 동결

model.summary()

import time
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
