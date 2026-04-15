#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import r2_score, accuracy_score


path = 'C:\\프로그램\\ai5\\_data\\kaggle\\santander\\'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop('target', axis=1)
y = train_csv['target']

print(x.shape)
print(y.shape)

# y = pd.get_dummies(y)
from sklearn.decomposition import PCA

pca = PCA()

print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=1542, stratify=y)


x = np.concatenate([x_train, x_test], axis=0)
from sklearn.decomposition import PCA
pca = PCA(n_components=13)
x = pca.fit_transform(x)
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print('0.95이상 : ', np.argmax(evr_cumsum>=0.95)+1)
print('0.99이상 : ', np.argmax(evr_cumsum>=0.99)+1)
print('0.999이상 : ', np.argmax(evr_cumsum>=0.999)+1)
print('0.1이상 : ', np.argmax(evr_cumsum>=1)+1)
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler


for i in range(64) : 
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
PCA : 1
r2 : -0.0006870116909527546
걸린 시간 : 29.95 초
===============================================
(140000, 1) (60000, 1)
(60000,) (60000, 1)
===============================================
결과 2
PCA : 1
r2 : -0.0007933631656842355
걸린 시간 : 38.28 초
===============================================
(140000, 1) (60000, 1)
(60000,) (60000, 1)
===============================================
결과 3
PCA : 1
r2 : -0.0002890339124319574
걸린 시간 : 40.85 초
===============================================
결과 4
PCA : 13
r2 : 0.005093346595080983
걸린 시간 : 62.57 초
===============================================
'''