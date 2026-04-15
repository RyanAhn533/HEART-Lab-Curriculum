from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time as t
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)

# 스케일링 적용
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# KFold 설정
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델 훈련 및 평가
r2_scores = []
losses = []

for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = Sequential()
    model.add(Dense(128, input_dim=10, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['acc'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, 
                       restore_best_weights=True)
    
    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=1,
        save_best_only=True, 
        filepath=("./_save/keras32/keras32_dropout.h1"))

    model.fit(x_train, y_train, epochs=200, batch_size=128, 
              verbose=0, validation_split=0.2, callbacks=[es])

    # 평가 및 예측
    loss = model.evaluate(x_test, y_test)
    losses.append(loss)
    y_predict = model.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    r2_scores.append(r2)
    
    print("로스는 ?", loss)
    print("r2스코어는? ", r2)

# KFold의 평균 결과 출력
print(f"\n최종 평균 로스: {np.mean(losses)}")
print(f"최종 평균 r2스코어: {np.mean(r2_scores)}")

'''
69/69 [==============================] - 0s 426us/step - loss: 99.1557 - acc: 0.0106
로스는 ? [99.15565490722656, 0.010560146532952785]
r2스코어는?  0.9968272981813956
69/69 [==============================] - 0s 309us/step - loss: 37.6631 - acc: 0.0087
로스는 ? [37.663124084472656, 0.008727606385946274]
r2스코어는?  0.9989042050227692
69/69 [==============================] - 0s 324us/step - loss: 166.8043 - acc: 0.0092
로스는 ? [166.8042755126953, 0.009186954237520695]
r2스코어는?  0.9948011427515466
69/69 [==============================] - 0s 324us/step - loss: 89.7843 - acc: 0.0101
로스는 ? [89.7843246459961, 0.010105649940669537]
r2스코어는?  0.9972774922588881
69/69 [==============================] - 0s 338us/step - loss: 257.8288 - acc: 0.0096
로스는 ? [257.8287658691406, 0.009646302089095116]
r2스코어는?  0.9922637352169528

최종 평균 로스: 65.12843716787174
최종 평균 r2스코어: 0.9960147746863104
'''