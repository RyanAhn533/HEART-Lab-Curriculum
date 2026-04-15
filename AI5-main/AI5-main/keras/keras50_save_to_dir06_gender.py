import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split

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
start_time=time.time()
np_path = 'c:/프로그램/ai5/_data/kaggle/Biggest_gender/'
x_train1=np.load(np_path + 'man_x_train1.npy')
y_train1=np.load(np_path + 'man_y_train1.npy')
x_test1=np.load(np_path + 'man_x_test.npy')
y_test1=np.load(np_path + 'man_y_test.npy')

x_train2=np.load(np_path + 'woman_x_train1.npy')
y_train2=np.load(np_path + 'woman_x_train2.npy')
x_test2=np.load(np_path + 'woman_y_train1.npy')
y_test2=np.load(np_path + 'woman_y_train2.npy')

# print(x_train1.shape)
# print(x_train1.shape)
# print(x_test1.shape)
# print(y_test1.shape)
# print(x_train2.shape)
# print(x_train2.shape)
# print(x_test2.shape)
# print(y_test2.shape)
train_datagen =  ImageDataGenerator(
    #rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)
augment_size = 8189

randidx = np.random.randint(x_train2.shape[0], size = augment_size)
x_augmented = x_train2[randidx].copy()
y_augmented = y_train2[randidx].copy()

x_augmented = x_augmented.reshape(
                                  x_augmented.shape[0], # 40000
                                  x_augmented.shape[1], #28
                                  x_augmented.shape[2], 3)
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir = 'C:\\프로그램\\ai5\\_data\\_save_img\\06'
).next()[0]


x_train = np.concatenate((x_train2,x_augmented), axis = 0)
print(x_train.shape)
y_train = np.concatenate((y_train2, y_augmented), axis = 0)
print(x_train.shape)
print(y_train.shape)
print(x_train1.shape)
print(x_train1.shape)


x_train = np.concatenate((x_train, x_train1),axis=1)
y_train = np.concatenate((y_train, y_train1),axis=1)
test_datagen = ImageDataGenerator(rescale=1./255)
path1 = "C:\\프로그램\\ai5\\_data\\image\\me\\"
xy_test = test_datagen.flow_from_directory(
    path1, target_size=(100, 100),
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
)
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, train_size=0.9, random_state=5656)

# #2. modeling
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=128, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dense(1024, activation='relu')) 
model.add(Dense(512, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))
                                          
                        
#3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras45\\k45_07\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k45_07_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ################### 

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
model.fit(x_train, y_train, epochs=30, batch_size=1, validation_split=0.2, callbacks=[es, mcp])
# model.fit_generator(x_train, y_train,
#                     epochs=1000,
#                     verbose=1,
#                     callbacks=[es, mcp],
#                     validation_steps=50)
end_time=time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = np.round(model.predict(xy_test, batch_size=1))
print("걸린 시간 :", round(end_time-start_time,2),'초')

# y_pre = np.round(y_pre)
