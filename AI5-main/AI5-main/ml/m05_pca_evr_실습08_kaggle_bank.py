from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import sklearn as sk
from sklearn.datasets import load_boston
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
from sklearn.preprocessing import LabelEncoder
###############

path = "./_data/kaggle/Bank/"

# CSV 파일 로드
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

# 데이터 확인
print(train_csv['Geography'].value_counts())

# 데이터 변환
train_csv['Geography'] = train_csv['Geography'].replace({'France': 1, 'Spain': 2, 'Germany': 3})
test_csv['Geography'] = test_csv['Geography'].replace({'France': 1, 'Spain': 2, 'Germany': 3})
train_csv['Gender'] = train_csv['Gender'].replace({'Male': 1, 'Female': 2})
test_csv['Gender'] = test_csv['Gender'].replace({'Male': 1, 'Female': 2})


# 특정 열에 0 값을 가진 행 삭제
"""
train_csv = train_csv[train_csv['Balance'] != 0]
test_csv = test_csv[test_csv['Balance'] != 0]
"""

# 문자열 값을 가진 열 확인 및 삭제
print(train_csv.select_dtypes(include=['object']).columns)
print(test_csv.select_dtypes(include=['object']).columns)

# 'Surname' 열 삭제
train_csv = train_csv.drop(['Surname'], axis=1)
test_csv = test_csv.drop(['Surname'], axis=1)

# 데이터 저장
train_csv.to_csv(path + "replaced_train.csv")
test_csv.to_csv(path + "replaced_test.csv")

# 데이터 로드
re_train_csv = pd.read_csv(path + "replaced_train.csv", index_col=0)
re_test_csv = pd.read_csv(path + "replaced_test.csv", index_col=0)

# 데이터 확인
re_train_csv.info()
re_test_csv.info()

# 특정 열 제거
re_train_csv = re_train_csv.drop(['CustomerId'], axis=1)
re_test_csv = re_test_csv.drop(['CustomerId'], axis=1)


# 데이터 스케일링
scaler = StandardScaler()
re_train_csv_scaled = scaler.fit_transform(re_train_csv.drop(['Exited'], axis=1))
re_test_csv_scaled = scaler.transform(re_test_csv)

"""
MaxAbsScaler
로스값은 :  0.32643696665763855
정확도는 :  0.8613021480292059
R2 스코어 : 0.16241584680842946

RobustScaler
로스값은 :  0.3278585970401764
정확도는 :  0.8619080801042203
R2 스코어 : 0.16607501742088715
"""

# 데이터프레임으로 변환
re_train_csv = pd.concat([pd.DataFrame(re_train_csv_scaled), re_train_csv['Exited'].reset_index(drop=True)], axis=1)
re_test_csv = pd.DataFrame(re_test_csv_scaled)



# 학습 데이터 분리
x = re_train_csv.drop(['Exited'], axis=1)
y = re_train_csv['Exited']
print(type(x))  # <class 'pandas.core.frame.DataFrame'>
print(type(y))  # <class 'pandas.core.series.Series'>

x = x.to_numpy()

print(re_train_csv.shape)
print(re_test_csv.shape)
print(type(x))

x = x.reshape(165034, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1186, train_size=0.8)
x = np.concatenate([x_train, x_test], axis=0)
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
x = pca.fit_transform(x)
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print('0.95이상 : ', np.argmax(evr_cumsum>=0.95)+1)
print('0.99이상 : ', np.argmax(evr_cumsum>=0.99)+1)
print('0.999이상 : ', np.argmax(evr_cumsum>=0.999)+1)
print('0.1이상 : ', np.argmax(evr_cumsum>=1)+1)
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler


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
PCA : 9
r2 : 0.3899280039829591
걸린 시간 : 48.97 초
===============================================
(132027, 10) (33007, 10)
(33007,) (33007, 1)
===============================================
결과 2
PCA : 10
r2 : 0.39357083894296074
걸린 시간 : 54.66 초
===============================================
(132027, 10) (33007, 10)
(33007,) (33007, 1)
===============================================
결과 3
PCA : 10
r2 : 0.390590088727895
걸린 시간 : 92.27 초
===============================================
'''