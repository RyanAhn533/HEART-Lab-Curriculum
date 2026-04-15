import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import random as rn

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)
# 데이터 경로 설정
path = "C:/프로그램/ai5/_data/kaggle/santander/"

# 데이터 로드
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 데이터 분리
x = train_csv.drop('target', axis=1)
y = train_csv['target']

# 데이터 스케일링
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, stratify=y)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


# 모델 생성


model = Sequential()
model.add(Dense(16))
model.add(Dense(1))

# 모델 컴파일 및 훈련
#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

for i in range(1): 
    learning_rate = [0.01]
    # learning_rate = 0.0007       # default = 0.001
    learning_rate = learning_rate[i]
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50, 
                   verbose=1, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', mode = 'auto', 
                        patience=25, verbose=1, factor=0.8)#running rate * factor)

    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1, verbose=0,
          batch_size=512,callbacks=[es, rlr]
          )

#4. 평가,예측
    print("=================1. 기본출력 ========================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 :{1}'.format(learning_rate, loss))

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))
   
    
'''
전
=================1. 기본출력 ========================
lr : 0.01, 로스 :0.08219298720359802
lr : 0.01, r2 : 0.09064908749237499
후
=================1. 기본출력 ========================
lr : 0.01, 로스 :0.08007656782865524
lr : 0.01, r2 : 0.11406413172384999


후
=================1. 기본출력 ========================
lr : 0.01, 로스 :0.08488960564136505
lr : 0.01, r2 : 0.060816027209383505
'''