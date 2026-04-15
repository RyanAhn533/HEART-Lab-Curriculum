from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img # = 이미지 땡겨와
from tensorflow.keras.preprocessing.image import img_to_array # 땡겨온거 수치화
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
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
path_train = 'C:\\프로그램\\ai5\\_data\\kaggle\\Biggest_gender\\faces'
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
)

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)
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
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
                                          
                        
#3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras47\\k47_07\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k47_07_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ################### 

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
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

y_pre = np.round(model.predict(x_test, batch_size=16))
print("걸린 시간 :", round(end_time-start_time,2),'초')

#acc : 
# 
# 0.58766