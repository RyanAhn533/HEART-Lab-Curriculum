#url : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data?select=test.csv
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from sklearn.model_selection import KFold, StratifiedKFold

path = "C:/프로그램/ai5/_data/kaggle/santander/"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop('target', axis=1)
y = train_csv['target']

# 스케일링 적용
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x = np.array(x)
y = np.array(y)
x = x.reshape(x.shape[0], 200, 1)
print(x.shape, y.shape)


# KFold 설정
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델 훈련 및 평가
r2_scores = []
losses = []
accuracys = []
for train_index, test_index in kfold.split(x,y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

#모델
    model = Sequential()
    model.add(Conv1D(filters=10, kernel_size=2, input_shape=(200, 1)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))
    
    

#컴파일 훈련
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20,
                  restore_best_weights=True)

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    mcp = ModelCheckpoint(monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras31_mcp/keras31_mcp_12_santander.h1"))

    model.fit(x_train, y_train, epochs=50, batch_size=128,
              verbose=1, validation_split=0.2, callbacks=[es])



#평가예측

    loss = model.evaluate(x_test, y_test)
    losses.append(loss)

    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)
    r2 = r2_score(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    accuracys.append(acc_score)

    print('로스값은 : ', loss)
    print('정확도는 : ', accuracy_score)
    print("R2 스코어 :", r2)

# KFold의 평균 결과 출력
print(f"\n최종 평균 로스: {np.mean(losses)}")
print(f"최종 평균 r2스코어: {np.mean(r2_scores)}")
print(f"최종 평균 r2스코어: {np.mean(accuracys)}")
'''
로스값은 :  [1.5498253107070923, 0.899524986743927]
정확도는 :  0.0
R2 스코어 : -7418.2196720672555
'''