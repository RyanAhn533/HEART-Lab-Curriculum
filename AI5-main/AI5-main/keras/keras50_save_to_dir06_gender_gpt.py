import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time
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
# 데이터 로드
np_path = 'c:/프로그램/ai5/_data/새 폴더/'
x_train1 = np.load(np_path + 'keras49_06_x_train_man.npy')
y_train1 = np.load(np_path + 'keras49_06_y_train_man.npy')

path = 'C:\\프로그램\\ai5\\_data\\image\\me\\'
x_test2 = np.load(path + 'me_x_train2.npy')

x_train2 = np.load(np_path + 'keras49_06_x_train_woman.npy')
y_train2 = np.load(np_path + 'keras49_06_y_train_woman.npy')

# 데이터 결합
x_train = np.concatenate((x_train1, x_train2), axis=0)
y_train = np.concatenate((y_train1, y_train2), axis=0)

print(x_train.shape)
print(y_train.shape)
# 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode="nearest"
)

# 데이터 증강 및 추가
augment_size = 5000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

x_train3 = np.concatenate((x_train, x_augmented), axis=0)
y_train3 = np.concatenate((y_train, y_augmented), axis=0)

x_train = np.concatenate((x_train3, x_train1), axis=0)
y_train = np.concatenate((y_train3, y_train1), axis=0)

# train/test 분할
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=42)

# 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(80, 80, 3), padding='same'))
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

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 콜백 설정
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)

date = datetime.datetime.now().strftime("%m%d_%H%M")
filepath = f'C:/ai5/_save/keras45/k45_07_{date}_{{epoch:04d}}-{{val_loss:.4f}}.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# 모델 학습
start_time = time.time()
model.fit(x_train, y_train, epochs=30, batch_size=10, validation_split=0.2, callbacks=[es, mcp])
end_time = time.time()

# 모델 평가
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))
y_predict = model.predict(x_test2)
# 학습 시간 출력
print("걸린 시간 :", round(end_time - start_time, 2), '초')
print(y_predict)