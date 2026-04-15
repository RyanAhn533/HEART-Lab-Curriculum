import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# 데이터 로드
x, y = load_breast_cancer(return_X_y=True)

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=4444, train_size=0.8, stratify=y)

# 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 모델 정의
xgb = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)  # verbose=0 추가하여 출력을 억제
rf = RandomForestClassifier()
model = StackingClassifier(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs=-1,
    cv=5
)

# 모델 훈련
model.fit(x_train, y_train)

# 평가 및 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('스태킹 ACC:', accuracy_score(y_test, y_pred))

# model.score :  0.9649122807017544
# 스태킹 ACC: 0.9649122807017544
