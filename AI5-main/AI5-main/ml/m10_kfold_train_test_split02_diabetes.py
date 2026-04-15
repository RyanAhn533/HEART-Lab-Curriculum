from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 데이터 로드
dataset = load_diabetes()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

# 2. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8)

# 3. 데이터 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. 모델 설정 - 회귀 모델로 변경
model = SVR()

# 5. KFold 설정
n_splits = 3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# 6. 교차 검증 점수 계산
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('R2 Scores:', scores)
print('Mean R2 Score:', '\n 평균 ACC', round(np.mean(scores), 4))

# 7. 교차 검증 예측
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

# 8. R2 Score 계산
r2 = r2_score(y_test, y_predict)
print('Cross-Val Predict R2 Score:', r2)
'''
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
R2 Scores: [0.03920164 0.12050794 0.11398491]
Mean R2 Score: 0.0912
Cross-Val Predict R2 Score: 0.0295448360202053
'''