import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os

# 1. 데이터 준비

# 경로 설정
path_train1 = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/train/'
path_train2 = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/test1/'
path_train3 = 'C:/프로그램/ai5/_data/image/cat_and_dog/train/'
path_train4 = 'C:/프로그램/ai5/_data/image/cat_and_dog/test/'

# ImageDataGenerator 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.2,
    rotation_range=15,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.2,
    rotation_range=15,
    fill_mode='nearest'
)
# 학습 데이터 로드
xy_train1 = train_datagen.flow_from_directory(
    path_train1, target_size=(80, 80), 
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
).next()

xy_test1 = test_datagen.flow_from_directory(
    path_train2, target_size=(80, 80),
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
).next()


np_path = 'C:\\프로그램\\ai5\\_data\\mixed_data'
np.save(np_path + 'cat_dog_kaggle_train.npy', arr=xy_train1)

np_path = 'C:\\프로그램\\ai5\\_data\\mixed_data'
np.save(np_path + 'cat_dog_kaggle_train.npy', arr=xy_test1)

xy_train2 = train_datagen.flow_from_directory(
    path_train3, target_size=(80, 80), 
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
).next()
xy_test2 = train_datagen.flow_from_directory(
    path_train4, target_size=(80, 80),
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)
np_path = 'C:\\프로그램\\ai5\\_data\\mixed_data'
np.save(np_path + 'cat_dog_kaggle_train.npy', arr=xy_train2)

np_path = 'C:\\프로그램\\ai5\\_data\\mixed_data'
np.save(np_path + 'cat_dog_kaggle_train.npy', arr=xy_test2)

"""
# 학습 데이터 병합
x_train = np.concatenate((xy_train1[0], xy_train2[0]), axis=0)
y_train = np.concatenate((xy_train1[1], xy_train2[1]), axis=0)

# 테스트 데이터를 위한 ImageDataGenerator 설정
test_datagen = ImageDataGenerator(rescale=1./255)

# 테스트 데이터 로드
xy_test1 = test_datagen.flow_from_directory(
    path_train1, target_size=(100, 100), 
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
).next()

xy_test2 = test_datagen.flow_from_directory(
    path_train2, target_size=(100, 100), 
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
).next()

# 테스트 데이터 병합
x_test = np.concatenate((xy_test1[0], xy_test2[0]), axis=0)
y_test = np.concatenate((xy_test1[1], xy_test2[1]), axis=0)

# 데이터 저장
np_path = 'C:/프로그램/ai5/_data/mixed_data/'
os.makedirs(np_path, exist_ok=True)
np.save(np_path + 'cat_dog_train.npy', arr=(x_train, y_train))
np.save(np_path + 'cat_dog_test.npy', arr=(x_test, y_test))
exit()
# 2. 모델 구성
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 3. 컴파일 및 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mcp = ModelCheckpoint(filepath='./best_model.h5', monitor='val_loss', save_best_only=True)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=32, callbacks=[es, mcp])

# 4. 평가 및 예측
loss, acc = model.evaluate(x_test, y_test)
print(f'Loss: {loss}, Accuracy: {acc}')
"""