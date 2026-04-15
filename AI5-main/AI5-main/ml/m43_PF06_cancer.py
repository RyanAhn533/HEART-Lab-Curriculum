import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# 1. 데이터 로드 및 전처리
x, y = load_breast_cancer(return_X_y=True)

# 다항식 변환
def preprocess_with_polynomial(x, degree):
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = pf.fit_transform(x)
    return x_poly

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 정규화
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 다항식 변환 후 데이터 준비
x_train_poly = preprocess_with_polynomial(x_train, degree=2)
x_test_poly = preprocess_with_polynomial(x_test, degree=2)

# 2. 모델 구성
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')




# 3. 성능 비교 함수 정의
def evaluate_model(model, x_train, x_test, y_train, y_test, label):
    # 모델 학습
    model.fit(x_train, y_train)
    # 예측 및 평가
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{label} Accuracy: {accuracy:.4f}")
    return accuracy

# x 성능 평가
print("Evaluating model with original x")
accuracy_x = evaluate_model(model, x_train, x_test, y_train, y_test, label="Original x")

# x_poly 성능 평가
print("Evaluating model with x_poly")
accuracy_x_poly = evaluate_model(model, x_train_poly, x_test_poly, y_train, y_test, label="Polynomial x")

print("그냥 x",accuracy_x,"x 뽈리", accuracy_x_poly)

# 그냥 x 0.956140350877193 x 뽈리 0.9824561403508771