"""
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
# path = "C:\\프로그램\\ai5\\_data\\image\\me\\2.jpg"
# img = load_img(path, target_size=(200, 200,))
# print(img)
# print(type(img))
# plt.imshow(img)
# plt.show()

# arr = img_to_array(img)
# print(arr)
# print(arr.shape) #(200, 200, 3)
# print(type(arr))
# img = np.expand_dims(arr, axis=0) #arr = arr.reshape(1,100,100,3)
# print(img.shape)
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

######################요기부터 중요########################
datagen = ImageDataGenerator(
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

it = datagen.flow(img, 
             batch_size=1,
             )
print(it)
print(it.next())
np_path = 'c:/프로그램/ai5/_data/image/me/'
np.save(np_path + 'me_x_train1.npy')
np.save(np_path + 'me_y_train1.npy')

"""
#flow_from_diecory는 이미지를 가져다가 증폭하거나 이것적서 그런 작업을 하는데
#flow 는 연합을 하거나 증폭을 한다
# fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))

# for i in range(5):
#     batch = it.next()
#     ax[i].imshow(batch)
#     ax[i].axis('off')
    
# plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

# 이미지 로드 및 변환
path = "C:\\프로그램\\ai5\\_data\\image\\me\\me1\\2.jpg"
img = load_img(path, target_size=(80, 80))
arr = img_to_array(img)
img = np.expand_dims(arr, axis=0) # arr = arr.reshape(1, 200, 200, 3)

# ImageDataGenerator 설정 및 데이터 증강
datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True, 
    vertical_flip=True, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    rotation_range=5, 
    zoom_range=1.2, 
    shear_range=0.7, 
    fill_mode="nearest"
)

# 증강된 이미지 생성
it = datagen.flow(img, batch_size=1)
batch = it.next()  # 배치 하나의 증강된 이미지 생성

# 이미지 저장 (예: 첫 번째 증강된 이미지 저장)
np_path = 'c:/프로그램/ai5/_data/image/me/'
np.save(np_path + 'me_x_train2.npy', batch[0])  # x 데이터 저장

# 증강된 이미지 시각화
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))
for i in range(5):
    batch = it.next()
    ax[i].imshow(batch[0])
    ax[i].axis('off')
    
plt.show()
