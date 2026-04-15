import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

np_path = 'C:\\프로그램\\ai5\\_data\\mixed_data\\'
x_train1=np.load(np_path + 'cat_dog_image_x_train.npy')
y_train1=np.load(np_path + 'cat_dog_image_y_train.npy')
x_train2=np.load(np_path + 'cat_dog_kaggle_x_train.npy')
y_train2=np.load(np_path + 'cat_dog_kaggle_y_train.npy')
x_test1=np.load(np_path + 'cat_dog_kaggle_x_test.npy')
print(x_train1.shape)
print(y_train1.shape)
print(x_train2.shape)
print(y_train2.shape)
print(x_test1.shape)

x = np.concatenate((x_train1,x_train2), axis = 0)
y = np.concatenate((y_train1, y_train2), axis = 0)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=5656)

train_datagen = ImageDataGenerator(
    rescale=1/255,

horizontal_flip=True, #수평 뒤집기
    vertical_flip=True, #수직뒤집기
    width_shift_range=0.1, #평행이동
    height_shift_range=0.1, #평행이동 수직
    rotation_range=1, #정해진 각도만큼 이미지 회전
    zoom_range=0.2, #축소 또는 확대
    shear_range=0.7, # 좌표
    fill_mode="nearest", #비율에 맞춰서 채워라
    )
start_time=time.time()

augment_size = 5000
randidx = np.random.randint(x_train.shape[0], size = augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)
print(y_augmented.shape)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
     save_to_dir = 'C:\\프로그램\\ai5\\_data\\_save_img\\05').next()[0]
# print(x_augmented.shape)
# print(x_train.shape)
# print(y_train.shape)



# x_train = np.concatenate((x_train,x_augmented), axis = 0)
# print(x_train.shape)
# y_train = np.concatenate((y_train, y_augmented), axis = 0)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)



# # #2. modeling
# model = Sequential()

# model.add(Conv2D(32, (3,3), activation='relu', input_shape=(80, 80, 3), padding='same')) 
# model.add(MaxPool2D())
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3), padding='same')) 
# model.add(MaxPool2D())
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=128, activation='relu', kernel_size=(3,3), padding='same')) 
# model.add(MaxPool2D())
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
# model.add(MaxPool2D())
# model.add(Dropout(0.25))

# model.add(Flatten()) 
# model.add(Dense(1024, activation='relu')) 
# model.add(Dense(512, activation='relu')) 
# model.add(Dense(1, activation='sigmoid'))

# # 모델 컴파일
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True)

# ################## mcp 세이브 파일명 만들기 시작 ###################
# import datetime
# date = datetime.datetime.now()
# print(date) #2024-07-26 16:49:57.565880
# print(type(date)) #<class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M")
# print(date) #0726_1654
# print(type(date)) #<class 'str'>


# path = 'C:\\ai5\\_save\\keras45\\k45_07\\'
# filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
# filepath = "".join([path, 'k45_07_', date, '_' , filename])
# #생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
# ################## mcp 세이브 파일명 만들기 끝 ################### 

# mcp=ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose = 1,
#     save_best_only=True,
#     filepath=filepath)

# model.fit(x_train, y_train, epochs=30, batch_size=16, validation_split=0.2, callbacks=[es, mcp])

# # 평가 예측
# loss = model.evaluate(x_test, y_test, verbose=1, batch_size=16)
# print('loss :', loss[0])
# print('acc :', round(loss[1],5))

# # 예측 값 생성 및 반올림
# y_pre = np.round(model.predict(x_test, batch_size=16))


# # 예측 값 처리
# y_submit = model.predict(x_test, batch_size=16)


