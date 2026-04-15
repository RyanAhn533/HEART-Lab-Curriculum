import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os

# 경로 설정
path_train1 = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/train/'
path_train2 = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/test1/'


# ImageDataGenerator 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.2,
    rotation_range=15,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255,) #테스트 데이터는 절대 변환하지않는다. 데이터 조작.

# 학습 데이터 로드
xy_train1 = train_datagen.flow_from_directory(
    path_train1, target_size=(80, 80), 
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    path_train2, target_size=(80, 80), #(10,200,200,1)->16개가 나옴 (batchsize, targetsize, channel) 
    #found 160 - xy_train -> batchsize 10 -> 16개있음
    batch_size=30000, 
#요 폴더에 있는걸 전부 수치화 해라
class_mode='binary',
color_mode='grayscale',
)



np_path = 'C:\\프로그램\\ai5\\_data\\mixed_data\\'
np.save(np_path + 'cat_dog_kaggle_x_train.npy', arr=xy_train1[0][0])
np.save(np_path + 'cat_dog_kaggle_y_train.npy', arr=xy_train1[0][1])
np.save(np_path + 'cat_dog_kaggle_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'cat_dog_kaggle_y_test.npy', arr=xy_test[0][1])

path_train3 = 'C:/프로그램/ai5/_data/image/cat_and_dog/Train/'
# ImageDataGenerator 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.2,
    rotation_range=15,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255,) #테스트 데이터는 절대 변환하지않는다. 데이터 조작.

# 학습 데이터 로드
xy_train3 = train_datagen.flow_from_directory(
    path_train3, target_size=(80, 80), 
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)


np_path = 'C:\\프로그램\\ai5\\_data\\mixed_data\\'

np.save(np_path + 'cat_dog_image_x_train.npy', arr=xy_train3[0][0])
np.save(np_path + 'cat_dog_image_y_train.npy', arr=xy_train3[0][1])