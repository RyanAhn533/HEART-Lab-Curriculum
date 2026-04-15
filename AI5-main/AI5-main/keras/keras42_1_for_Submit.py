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


# 모델 구성
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

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 콜백 설정
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_data/kaggle/dogs_vs_cats/saved_model/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'save_model', date, '_', filename])

mcp = ModelCheckpoint(
    monitor= 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only= True,
    filepath = filepath
)

# 모델 훈련
model.fit(x_train, y_train, 
          epochs = 100, 
          batch_size = 8, 
          verbose=1, 
          validation_split=0.2, 
          callbacks=[es, mcp])

# 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :', round(loss[1],3))

# 예측 값 처리
y_predict = model.predict(xy_test[0][0])  # 확률 값을 이진 클래스로 변환
accuracy = accuracy_score(y_test, y_predict)
print('acc_score : ', accuracy)

# 제출 파일 작성
y_submit = model.predict(xy_test[0][0])
sampleSubmission = pd.read_csv('C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/sample_submission.csv', index_col=0)
sampleSubmission['label'] = y_submit
sampleSubmission.to_csv('C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/samplesubmission_0804_6.csv')
# > 0.5).astype("int32")