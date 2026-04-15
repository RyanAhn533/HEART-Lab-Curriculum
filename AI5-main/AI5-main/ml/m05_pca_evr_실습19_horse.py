#배치를 100으로 잡고
#x, y를 추출해서 모델을 맹그러봐
#acc 0.99이상

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout,LSTM, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split

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


train_datagen = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/image/horse_human/'


xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
class_mode='binary',

color_mode='rgb',
shuffle=True)


x_train,x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
x_train = x_train.reshape(718,100*100*3)
x_test = x_test.reshape(309,100*100*3)

x = np.concatenate([x_train, x_test], axis=0)
from sklearn.decomposition import PCA
pca = PCA(n_components=100*3)   # 4개의 컬럼이 3개로 바뀜
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율
print(evr)

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum>=0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum>=0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum>=1.0)+1)

# 0.95 이상 : 154
# 0.99 이상 : 331
# 0.999 이상 : 486
# 1.0 일 때 : 713
x = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]

for i in range(4) :

    pca = PCA(n_components = x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    model = Sequential()
    model.add(Dense(100, input_shape=(x[i],)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))



    model.summary()
    start = time.time()
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    model.fit(x_train1, y_train, epochs=50, batch_size=16)

    end = time.time()
    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    y_pre = model.predict(x_test1)
    print(x_test.shape, y_pre.shape)


    print(x_test.shape, y_pre.shape)
    print("===============================================")
    print('결과', i+1)
    print('PCA :',x[i])
    print('acc :', round(loss[1],8))
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")
'''
===============================================
결과 1
PCA : 298
acc : 0.89320385
걸린 시간 : 5.09 초
===============================================
===============================================
결과 2
PCA : 1
acc : 0.61488676
걸린 시간 : 5.43 초
===============================================
===============================================
결과 3
PCA : 1
acc : 0.592233
걸린 시간 : 6.57 초
===============================================
===============================================
결과 4
PCA : 300
acc : 0.96116507
걸린 시간 : 6.74 초
===============================================
'''