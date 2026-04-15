from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import time as t

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

# 학습 데이터 분리
x = re_train_csv.drop(['Exited'], axis=1)
y = re_train_csv['Exited']
print(type(x))  # <class 'pandas.core.frame.DataFrame'>
print(type(y))  # <class 'pandas.core.series.Series'>

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1186, train_size=0.8)

# 데이터 스케일링
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
re_test_scaled = scaler.transform(re_test_csv)

# 모델 구성
model = Sequential([
    Dense(16, input_dim=x_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 컴파일 및 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=200, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
start = t.time()
model.fit(x_train_scaled, y_train, epochs=1000, batch_size=256, verbose=1, validation_split=0.3, callbacks=[es])
end = t.time()

# 평가 및 예측
loss, accuracy = model.evaluate(x_test_scaled, y_test, verbose=1)
y_pred = model.predict(x_test_scaled)
y_pred_rounded = np.round(y_pred)

# R2 및 정확도 계산
r2 = r2_score(y_test, y_pred_rounded)
accuracy = accuracy_score(y_test, y_pred_rounded)

# 예측 결과
y_submit = model.predict(re_test_scaled)
y_submit = np.round(y_submit)

print('로스값은 : ', loss)
print('정확도는 : ', accuracy)
print("R2 스코어 :", r2)
print("걸린 시간 :", round(end-start, 2), '초')

# 제출 파일 생성
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
sample_submission_csv['Exited'] = y_submit
sample_submission_csv.to_csv(path + "submission_0723_6.csv")

"""
로스값은 :  0.327251136302948
정확도는 :  0.861938376707971
R2 스코어 : 0.16625797595151004
걸린 시간 : 45.78 초
"""