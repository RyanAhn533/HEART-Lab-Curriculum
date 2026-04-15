import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, RandomForestRegressor


#LogisticRegression -> 분류 모델
#삼성 기술면접에서 logisticRegression이 분류 !

x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=4444, train_size=0.8,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
#model = DecisionTreeClassifier()
#model = BaggingRegressor(DecisionTreeRegressor(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=5656,
#                           bootstrap=True, #디폴트, 중복허용
                           #bootstrap = False, #중복허용 안함
#                           )
#model = BaggingRegressor(DecisionTreeRegressor(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=False, )#디폴트, 중복허용
                          #bootstrap = False, #중복허용 안함
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('엒뀨라씨 스코어', r2)

'''
1. bagging DecisionTree
최종점수 :  0.8107434957086668
엒뀨라씨 스코어 0.8107434957086668

2. Bagging Decision bootstrap=False
최종점수 :  0.6302361114247275
엒뀨라씨 스코어 0.6302361114247275

3. RandomRegressor
최종점수 :  0.8136010361205513
엒뀨라씨 스코어 0.8136010361205513
'''