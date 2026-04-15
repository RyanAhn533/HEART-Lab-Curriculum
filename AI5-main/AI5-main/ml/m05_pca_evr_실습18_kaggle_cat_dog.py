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
from sklearn.decomposition import PCA

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
x_train = x_train.reshape(25000,100*100*3)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=1186)
print(x_train)

x = np.concatenate([x_train, x_test], axis=0)
pca = PCA(n_components=28*28)   # 4개의 컬럼이 3개로 바뀜
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율
print(evr)

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum >= 1.0)+1)

# 0.95 이상 : 154
# 0.99 이상 : 331
# 0.999 이상 : 486
# 1.0 일 때 : 713
x = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1,
     np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]
for i in range(4) :
    pca = PCA(n_components = x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    model = Sequential()
    model.add(Dense(100, input_shape=(x[i],)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))



    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=0, 
                   restore_best_weights=True)

##### mcp 세이브 파일명 만들기 #####
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = 'C:\\프로그램\\ai5\\_save\\ml04\\'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'ml04_', str(i+1), '_', date, '_', filename])  

    mcp = ModelCheckpoint(
            monitor='val_loss',
            mode='auto',
            verbose=0,     
            save_best_only=True,   
            filepath=filepath,)
    import time

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=5, batch_size=128,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es, mcp],
            )
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
PCA : 754
acc : 0.546
걸린 시간 : 2.13 초
===============================================
===============================================
결과 2
PCA : 1
acc : 0.53240001
걸린 시간 : 2.18 초
===============================================
===============================================
결과 2
PCA : 1
acc : 0.53240001
걸린 시간 : 2.18 초
===============================================
'''