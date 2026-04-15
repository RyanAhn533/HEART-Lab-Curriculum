import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time

#1 데이터

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

y_test = y_test.to_numpy()


x = np.concatenate([x_train, x_test], axis=0)
from sklearn.decomposition import PCA
pca = PCA(n_components=54)
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
    model.add (Dense(512, input_shape=(x[i],)))
    model.add (Dense(256, activation='relu'))
    model.add (Dense(256, activation='relu'))
    model.add (Dense(10, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
    hist = model.fit(x_train1, y_train, epochs=200, batch_size=2048,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es, mcp],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=1)

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
    
    
'''
===============================================
결과 1
PCA : 333
acc : 0.97039998
걸린 시간 : 10.07 초
===============================================
(60000, 545) (10000, 545)
313/313 [==============================] - 0s 1ms/step - loss: 0.1314 - acc: 0.9676
(10000, 784) (10000, 10)
(10000, 784) (10000, 1)
===============================================
결과 2
PCA : 545
acc : 0.96759999
걸린 시간 : 11.58 초
===============================================
(60000, 683) (10000, 683)
313/313 [==============================] - 0s 1ms/step - loss: 0.1462 - acc: 0.9625
(10000, 784) (10000, 10)
(10000, 784) (10000, 1)
===============================================
결과 3
PCA : 683
acc : 0.96249998
걸린 시간 : 10.93 초
===============================================
(60000, 713) (10000, 713)
313/313 [==============================] - 0s 1ms/step - loss: 0.1321 - acc: 0.9668
(10000, 784) (10000, 10)
(10000, 784) (10000, 1)
===============================================
결과 4
PCA : 713
acc : 0.96679997
걸린 시간 : 10.89 초
===============================================
'''