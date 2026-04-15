import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

for i in range(6): 
    learning_rate = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # learning_rate = 0.0007       # default = 0.001
    learning_rate = learning_rate[i]
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1, verbose=0,
          batch_size=512,
          )

#4. 평가,예측
    print("=================1. 기본출력 ========================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 :{1}'.format(learning_rate, loss))

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))
   
    
'''
=================1. 기본출력 ========================
lr : 0.1, 로스 :0.0770043358206749
lr : 0.1, r2 : 0.14805467431077823
=================1. 기본출력 ========================
lr : 0.01, 로스 :0.07604069262742996
lr : 0.01, r2 : 0.15871551663180616
=================1. 기본출력 ========================
lr : 0.005, 로스 :0.0769612193107605
lr : 0.005, r2 : 0.14853162256796726
=================1. 기본출력 ========================
lr : 0.001, 로스 :0.07441693544387817
lr : 0.001, r2 : 0.17668000901879988
=================1. 기본출력 ========================
lr : 0.0005, 로스 :0.07423724234104156
lr : 0.0005, r2 : 0.17866873066449074
=================1. 기본출력 ========================
lr : 0.0001, 로스 :0.07415685057640076
lr : 0.0001, r2 : 0.17955834999794074
'''