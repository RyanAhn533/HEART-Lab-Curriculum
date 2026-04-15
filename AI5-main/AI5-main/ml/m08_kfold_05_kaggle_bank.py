from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping
import time as t
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 파일 경로 설정
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


# 데이터프레임으로 변환
re_train_csv = pd.concat([pd.DataFrame(re_train_csv_scaled), re_train_csv['Exited'].reset_index(drop=True)], axis=1)
re_test_csv = pd.DataFrame(re_test_csv_scaled)



# 학습 데이터 분리
x = re_train_csv.drop(['Exited'], axis=1)
y = re_train_csv['Exited']
print(type(x))  # <class 'pandas.core.frame.DataFrame'>
print(type(y))  # <class 'pandas.core.series.Series'>
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
accuracys = []
for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = Sequential([
        Dense(16, input_dim=x_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    start = t.time()
    end = t.time()
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, 
                       restore_best_weights=True)
    
    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=1,
        save_best_only=True, 
        filepath=("./_save/ml08/bank.h5"))

    model.fit(x_train, y_train, epochs=100, batch_size=512, verbose=0, validation_split=0.3, callbacks=[es])

   # 평가 및 예측
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    y_pred = model.predict(x_test)
    y_pred_rounded = np.round(y_pred)

# R2 및 정확도 계산
    r2 = r2_score(y_test, y_pred_rounded)
    r2_scores.append(r2)
    accuracy = accuracy_score(y_test, y_pred_rounded)
    accuracys.append(accuracy)
    loss = model.evaluate(x_test, y_test)
    losses.append(loss)
# 예측 결과
    y_submit = model.predict(re_test_csv)
    y_submit = np.round(y_submit)

    print('로스값은 : ', loss)
    print('정확도는 : ', accuracy)
    print("R2 스코어 :", r2)
    print("걸린 시간 :", round(end-start, 2), '초')

# KFold의 평균 결과 출력
print(f"\n최종 평균 로스: {np.mean(losses)}")
print(f"최종 평균 r2스코어: {np.mean(r2_scores)}")
print(f"최종 평균 r2스코어: {np.mean(accuracys)}")


# 제출 파일 생성
'''
로스값은 :  [0.3288825452327728, 0.8629987835884094]
정확도는 :  0.8629987578392462
R2 스코어 : 0.17380257612029404
걸린 시간 : 0.0 초
1032/1032 [==============================] - 0s 309us/step - loss: 0.3278 - accuracy: 0.8616
1032/1032 [==============================] - 0s 334us/step - loss: 0.3278 - accuracy: 0.8616
로스값은 :  [0.32781535387039185, 0.8616353869438171]
정확도는 :  0.8616354106704638
R2 스코어 : 0.1707518700926367
걸린 시간 : 0.0 초
1032/1032 [==============================] - 0s 319us/step - loss: 0.3287 - accuracy: 0.8632
1032/1032 [==============================] - 0s 355us/step - loss: 0.3287 - accuracy: 0.8632
로스값은 :  [0.32866647839546204, 0.8632411360740662]
정확도는 :  0.863241130669252
R2 스코어 : 0.17770167788075886
걸린 시간 : 0.0 초
1032/1032 [==============================] - 0s 323us/step - loss: 0.3296 - accuracy: 0.8605
1032/1032 [==============================] - 0s 306us/step - loss: 0.3296 - accuracy: 0.8605
로스값은 :  [0.32963213324546814, 0.8604538440704346]
정확도는 :  0.8604538431241858
R2 스코어 : 0.1731395762539336
걸린 시간 : 0.0 초
1032/1032 [==============================] - 0s 310us/step - loss: 0.3282 - accuracy: 0.8623
1032/1032 [==============================] - 0s 330us/step - loss: 0.3282 - accuracy: 0.8623
로스값은 :  [0.3281939625740051, 0.862297773361206]
정확도는 :  0.8622977640429013
R2 스코어 : 0.17222229754383211
걸린 시간 : 0.0 초

최종 평균 로스: 0.5953817397356034
최종 평균 r2스코어: 0.17352359957829105
최종 평균 r2스코어: 0.8621253812692098
'''
