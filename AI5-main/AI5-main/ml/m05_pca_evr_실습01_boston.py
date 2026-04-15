from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import time as t
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier  # 분류는 classifiaer, 회귀는 regress
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling2D, LSTM
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical


##############
# 데이터 로드 및 전처리
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)

# 데이터 스케일링
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 훈련 및 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

x = np.concatenate([x_train, x_test], axis=0)
pca = PCA(n_components=13)
x = pca.fit_transform(x)
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

print('0.95이상 : ', np.argmax(evr_cumsum>=0.95)+1)
print('0.99이상 : ', np.argmax(evr_cumsum>=0.99)+1)
print('0.999이상 : ', np.argmax(evr_cumsum>=0.999)+1)
print('0.1이상 : ', np.argmax(evr_cumsum>=1)+1)
#0.95이상 :  9
#0.99이상 :  12
#0.999이상 :  13
#0.1이상 :  13


for i in range(3) : 
    x = [9, 12, 13]
    pca = PCA(n_components= x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    model = Sequential()
    model.add(Dense(100, input_shape=(x[i],)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))


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
    hist = model.fit(x_train1, y_train, epochs=5000, batch_size=128,
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
    print(x_test.shape, y_predict.shape)
    print("===============================================")
    print('결과', i+1)
    print('PCA :',x[i])
    print('r2 :', r2)
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")
    
'''    
===============================================
결과 1
PCA : 9
r2 : 0.883515238378828
걸린 시간 : 3.79 초
===============================================
(102,) (102, 1)
(102, 13) (102, 1)
===============================================
결과 2
PCA : 12
r2 : 0.8988635624708993
걸린 시간 : 2.98 초
===============================================
(102,) (102, 1)
(102, 13) (102, 1)
===============================================
결과 3
PCA : 13
r2 : 0.905580579926834
걸린 시간 : 7.23 초
===============================================
'''