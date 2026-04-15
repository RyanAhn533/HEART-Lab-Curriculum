import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from xgboost import XGBClassifier, XGBRFRegressor
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier



# 1. 데이터 로드 및 전처리
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target

print(df.info())  # 결측치 확인
print(df.describe())  # 데이터 요약

# x, y 분리
x = df.drop(['target'], axis=1)
y = df['target']

# 다항식 변환 함수
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

# 다항식 변환
x_train_poly = preprocess_with_polynomial(x_train, degree=2)
x_test_poly = preprocess_with_polynomial(x_test, degree=2)

# 2. 모델 구성
model = XGBRFRegressor(random_state=42, n_estimators=100)

# 3. 성능 비교 함수 정의
def evaluate_model(model, x_train, x_test, y_train, y_test, label):
    # 모델 학습
    model.fit(x_train, y_train)
    # 예측 및 평가
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{label} MSE: {mse:.4f}, R2: {r2:.4f}")
    return mse, r2

# x 성능 평가
print("Evaluating model with original x")
mse_x, r2_x = evaluate_model(model, x_train, x_test, y_train, y_test, label="Original x")

# x_poly 성능 평가
print("Evaluating model with x_poly")
mse_x_poly, r2_x_poly = evaluate_model(model, x_train_poly, x_test_poly, y_train, y_test, label="Polynomial x")

print(f"Original x -> MSE: {mse_x:.4f}, R2: {r2_x:.4f}")
print(f"Polynomial x -> MSE: {mse_x_poly:.4f}, R2: {r2_x_poly:.4f}")

# Evaluating model with original x
# Original x MSE: 0.2563, R2: 0.8044
# Evaluating model with x_poly
# Polynomial x MSE: 0.2459, R2: 0.8123
# Original x -> MSE: 0.2563, R2: 0.8044
# Polynomial x -> MSE: 0.2459, R2: 0.8123
