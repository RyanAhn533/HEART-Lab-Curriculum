import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingClassifier, StackingClassifier, StackingRegressor

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=4444, train_size=0.8,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
xgb = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor()
#model = XGBClassifier()

model = StackingRegressor(
    estimators=[('XGB', xgb), ('CAT', cat)],
    #voting='hard', #디폴트 
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가에측
results = model.score(x_test, y_test)
print('최종점수 :', results)

y_predict = model.predict(x_test)
# 회귀 문제에 적합한 지표 사용
r2 = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)

print("r2 score :", r2)
print("mae : ", mae)

# 최종점수 : 0.9036946687584658
# r2 score : 0.9036946687584658
# mae :  0.06895153565157158