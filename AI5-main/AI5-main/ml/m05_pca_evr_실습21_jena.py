# jena를 Dnn으로 구성
# x : (42만, 144, 144) -> 42만, (144*144)
# y " (42만, 144)"

#https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time 
from sklearn.model_selection import train_test_split
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

path = ".\\_data\\kaggle\\jena\\"

a3 = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
b4 = a3.tail(144)
b5 = b4['T (degC)']
a = a3.head(420407)


a1 = a.drop(['T (degC)'], axis=1)
a2 = a['T (degC)']

def split_x(dataset, size) :
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
        
    return np.array(aaa)
size = 144
x = split_x(a1, size)
y = split_x(a2, size)

x= np.delete(x, -1 , axis = 0)
# print("x변환 :", x[-1])
print(x.shape) #(420263, 144, 13)

y= np.delete(y, 0 , axis = 0)
# print("y변환 : ",y[0])
print(y.shape) #(420263, 144)
x = x.reshape(420263,144*13)
b = a.tail(144)

b1 = b.drop(['T (degC)'], axis=1)
b2 = b['T (degC)']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=5656)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 


x = np.concatenate([x_train, x_test], axis=0)
from sklearn.decomposition import PCA

pca = PCA(n_components=144)   # 4개의 컬럼이 3개로 바뀜
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
x = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]

for i in range(4) :

    pca = PCA(n_components = x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    model = Sequential()
    model.add(Dense(300, input_shape=(x[i],)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(144, activation='softmax'))



    
    start = time.time()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=0, 
                   restore_best_weights=True)
    model.fit(x_train1, y_train, epochs=50, batch_size=10, callbacks=[es], verbose=0)

    end = time.time()
    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    y_pre = model.predict(x_test1)
    print(x_test.shape, y_pre.shape)

    y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
    y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)
    print(x_test.shape, y_pre.shape)
    print("===============================================")
    print('결과', i+1)
    print('PCA :',x[i])
    print('acc :', round(loss[1],8))
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")
    
<<<<<<< HEAD
    #깃ㅇㄴㅁ
=======
    #깃깃깃
>>>>>>> 39f6337bcbe221709ca9fa6ffa820a5623171801
