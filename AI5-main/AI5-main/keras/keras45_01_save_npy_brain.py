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


test_datagen = ImageDataGenerator(
    rescale=1./255,) #테스트 데이터는 절대 변환하지않는다. 데이터 조작.
path_train = 'C:/프로그램/ai5/_data/image/brain/train/'
path_test =   'C:/프로그램/ai5/_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_test, target_size=(200, 200), #(10,200,200,1)->16개가 나옴 (batchsize, targetsize, channel) 
    #found 160 - xy_train -> batchsize 10 -> 16개있음
    batch_size=10, 
#요 폴더에 있는걸 전부 수치화 해라
class_mode='binary',

color_mode='grayscale',
shuffle=True)
xy_test = test_datagen.flow_from_directory(
    path_test, target_size=(200, 200), #(10,200,200,1)->16개가 나옴 (batchsize, targetsize, channel) 
    #found 160 - xy_train -> batchsize 10 -> 16개있음
    batch_size=10, 
#요 폴더에 있는걸 전부 수치화 해라
class_mode='binary',

color_mode='grayscale',
)

np_path = 'C:/프로그램/ai5/_data/image/brain/save_npy/'
np.save(np_path + 'keras45_01_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_01_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras45_01_x_test.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_01_y_test.npy', arr=xy_train[0][1])