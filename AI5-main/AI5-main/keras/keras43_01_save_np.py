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
path_train = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/train/'
path_test = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/test1/'

# 데이터 로드
xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    path_test, target_size=(100, 100),
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

start = time.time()

np_path = 'C:/프로그램/ai5/_data/_save/save_npy/'
np.save(np_path + 'keras43_01_xy_test.npy', arr=xy_train)

end = time.time()
print('걸린시간 : ', end-start)
