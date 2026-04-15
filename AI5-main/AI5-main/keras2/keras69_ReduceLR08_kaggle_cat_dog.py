import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten,LSTM, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
import tensorflow as tf
import tensorflow as tf
import random as rn

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)
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


start = time.time()

np_path = 'C:\\프로그램\\ai5\\_data\\_save\\save_npy\\'

x_train = np.load(np_path + "keras43_01_x_train1.npy")
y_train = np.load(np_path + "keras43_01_y_train1.npy")
x_test = np.load(np_path + "keras43_01_x_test.npy")
y_test = np.load(np_path + "keras43_01_y_test.npy")
x_train = x_train.reshape(25000,100*100,3)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=1186)
print(x_train)
print(x_train.shape)


model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(10000, 3)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
model.summary()

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
rlr = ReduceLROnPlateau(monitor='val_loss', mode = 'auto', 
                        patience=25, verbose=1, factor=0.8)#running rate * factor)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=20,
    restore_best_weights=True
)

# 평가 예측
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=16, callbacks=[es, rlr])
print('loss :', loss[0])
print('acc :', round(loss[1],5))

# 예측 값 생성 및 반올림
y_pre = np.round(model.predict(x_test, batch_size=16))

end_time = time.time()
print("걸린 시간 :", round(end_time-start,2),'초')


# 예측 값 처리
y_submit = model.predict(x_test, batch_size=16)

#loss : 0.6932843327522278
#acc : 0.4828
#걸린 시간 : 28.09 초

#loss : 7.360466957092285
#acc : 0.5196
#걸린 시간 : 6.95 초