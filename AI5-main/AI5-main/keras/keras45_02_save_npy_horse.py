#배치를 100으로 잡고
#x, y를 추출해서 모델을 맹그러봐
#acc 0.99이상

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
class_mode='binary',

color_mode='rgb',
)


x_train,x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)


np_path = 'C:/프로그램/ai5/_data/image/horse_human/save_npy/'
np.save(np_path + 'keras45_02_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_02_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras45_02_x_test.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_02_y_test.npy', arr=xy_train[0][1])