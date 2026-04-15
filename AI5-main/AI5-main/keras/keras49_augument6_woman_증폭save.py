#https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split

start_time=time.time()
np_path = 'c:/프로그램/ai5/_data/kaggle/biggest_gender/'
x_train1=np.load(np_path + 'woman_x_train1.npy')
y_train1=np.load(np_path + 'woman_x_train2.npy')
x_test1=np.load(np_path + 'woman_y_train1.npy')
y_test1=np.load(np_path + 'woman_y_train2.npy')
print(x_train1.shape)
print(y_train1.shape)


x_train= np.concatenate((x_train1,y_train1))
x_test= np.concatenate((x_test1,y_test1))
print(x_test.shape)
print(x_train.shape)

end_time=time.time()
print("데이터 불러오는 시간 :", round(end_time-start_time,2),'초') 

x_train, x_test, y_train, y_test = train_test_split(x_train, x_test, train_size=0.9, random_state=5656)

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
start_time=time.time()
augment_size = 5000
randidx = np.random.randint(x_train.shape[0], size = augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,).next()[0]
x_train = np.concatenate((x_train,x_augmented), axis = 0)
print(x_train.shape)
y_train = np.concatenate((y_train, y_augmented), axis = 0)
print(x_train.shape)
print(x_test.shape)
print(x_train.shape)

np_path = 'c:/프로그램/ai5/_data/kaggle/biggest_gender/'
np.save(np_path + 'bigwoman_x_train1.npy', arr=x_train)
np.save(np_path + 'bigwoman_y_train1.npy', arr=y_train)

