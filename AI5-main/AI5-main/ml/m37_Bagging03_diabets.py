import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, RandomForestRegressor


#LogisticRegression -> 분류 모델
#삼성 기술면접에서 logisticRegression이 분류 !

x, y = load_diabetes(return_X_y=True)

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
model = BaggingRegressor(DecisionTreeRegressor(),
                           n_estimators=100,
                           n_jobs=-1,
                           random_state=4444,
                           bootstrap=False, )#디폴트, 중복허용
                          #bootstrap = False, #중복허용 안함
#model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('엒뀨라씨 스코어', r2)

'''
랜담뽀라스트
최종점수 :  0.4821836059991692
엒뀨라씨 스코어 0.4821836059991692

배깅 디씨쟌 리그레써 bootstrap=True
최종점수 :  0.49530071144036325
엒뀨라씨 스코어 0.49530071144036325

배깅 디씨쟌 리그레써 bootstrap=False
최종점수 :  -0.18342750888969794
엒뀨라씨 스코어 -0.18342750888969794
'''