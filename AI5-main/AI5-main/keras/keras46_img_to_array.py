from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img # = 이미지 땡겨와
from tensorflow.keras.preprocessing.image import img_to_array # 땡겨온거 수치화
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, MaxPool2D

"""
path = "C:\\프로그램\\ai5\\_data\\image\\me\\2.jpg"
img = load_img(path, target_size=(200, 200,))
print(img)
print(type(img))
plt.imshow(img)
plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape) #(200, 200, 3)
print(type(arr))
img = np.expand_dims(arr, axis=0) #arr = arr.reshape(1,100,100,3)
print(img.shape)
"""
path = "C:\\프로그램\\ai5\\_data\\image\\me\\me1\\2.jpg"
img = load_img(path, target_size=(200, 200,))
print(img)
print(type(img))
plt.imshow(img)
plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape) #(200, 200, 3)
print(type(arr))
img = np.expand_dims(arr, axis=0) #arr = arr.reshape(1,100,100,3)
print(img.shape)
path_train = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/train/'
# 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=True, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    rotation_range=1, 
    zoom_range=0.2, 
    shear_range=0.7, 
    fill_mode="nearest"
)

# 테스트 데이터 증강 설정
test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 경로 설정


# 데이터 로드
xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)
path1 = "C:\\프로그램\\ai5\\_data\\image\\me\\"
xy_test = test_datagen.flow_from_directory(
    path1, target_size=(100, 100),
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

np.save(path1 + 'keras45_01_x_test.npy', arr=xy_train[0][0])
np.save(path1 + 'keras45_01_y_test.npy', arr=xy_train[0][1])