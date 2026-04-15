from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import sklearn as sk
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
##############
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

###############

path = 'C:\\프로그램\\ai5\\_data\\bike-sharing-demand\\'
# \a  \b 이런걸 하나의 문자로 인식함 줄바꿈 이런거
# # 위와같은 애들 \ -> \\로 바꿔줘야함 / // 도 가능
# path = 'C:/프로그램//ai5\_data\\bike-sharing-demand'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

train_csv = train_csv.dropna() #train_csv 데이터에서 결측치 삭제
test_csv = test_csv.fillna(test_csv.mean()) #test_csv에는 결측치 평균으로 넣기

x = train_csv.drop(['count'], axis = 1)
y = train_csv[['count']] #, 'registered


from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler

scaler = StandardScaler()
x = scaler.fit_transform(x)
print(x.shape)
x = x.reshape(10886,10)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    shuffle=True, random_state=3)
x = np.concatenate([x_train, x_test], axis=0)
pca = PCA(n_components=10)
x = pca.fit_transform(x)
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print('0.95이상 : ', np.argmax(evr_cumsum>=0.95)+1)
print('0.99이상 : ', np.argmax(evr_cumsum>=0.99)+1)
print('0.999이상 : ', np.argmax(evr_cumsum>=0.999)+1)
print('0.1이상 : ', np.argmax(evr_cumsum>=1)+1)
# 8 9 10 10
for i in range(4) : 
    x = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]
    pca = PCA(n_components= x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    print(x_train1.shape, x_test1.shape)
    
    
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
#scaler = MaxAbsScaler()
#scaler = StandardScaler()

#scaler = MaxAbsScaler() 0.61 r2 0.49
#scaler = RobustScaler() r2 0.304  loss 0.91951
#모델
    model = Sequential()
    model.add (Dense(128, input_shape=(x[i],)))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(64, activation='relu'))
    model.add (Dense(64, activation='relu'))
    model.add (Dense(32, activation='relu'))
    model.add (Dense(32, activation='relu'))
    model.add (Dense(1))



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
    hist = model.fit(x_train1, y_train, epochs=500, batch_size=128,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es, mcp],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)
    y_predict = model.predict(x_test1)
    print(y_test.shape, y_predict.shape)
    
    
    
    
    r2 = r2_score(y_test, y_predict)
    print("===============================================")
    print('결과', i+1)
    print('PCA :',x[i])
    print('r2 :', r2)
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")
    '''
    ===============================================
결과 1
PCA : 8
r2 : 0.9998785798407698
걸린 시간 : 9.34 초
===============================================
(8708, 9) (2178, 9)
(2178, 1) (2178, 1)
(2178, 10) (2178, 1)
===============================================
결과 2
PCA : 9
r2 : 0.9999743592039393
걸린 시간 : 15.55 초
===============================================
(8708, 10) (2178, 10)
(2178, 1) (2178, 1)
(2178, 10) (2178, 1)
===============================================
결과 3
PCA : 10
r2 : 0.9999664449652984
걸린 시간 : 13.91 초
===============================================
(8708, 10) (2178, 10)
(2178, 1) (2178, 1)
(2178, 10) (2178, 1)
===============================================
결과 4
PCA : 10
r2 : 0.9999754663663101
걸린 시간 : 17.88 초
===============================================
    
'''