#가장 대표적인게 Randomforest - decision tree를 bagging 시킨놈
#앙상블 기법 !
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier


#LogisticRegression -> 분류 모델
#삼성 기술면접에서 logisticRegression이 분류 !

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=4444, train_size=0.8,
                                                    stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
#model = DecisionTreeClassifier()
# model = BaggingClassifier(DecisionTreeClassifier(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=5656,
#                           bootstrap=True, #디폴트, 중복허용
#                           #bootstrap = False, #중복허용 안함
# #                           )
# model = BaggingClassifier(DecisionTreeClassifier(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=False, #디폴트, 중복허용
                          #bootstrap = False, #중복허용 안함
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('엒뀨라씨 스코어', acc)
#DecisionTree
#최종점수 :  0.9473684210526315
#엒뀨라씨 스코어 0.9473684210526315

#bagging DecisionTree bootstrap=True
# 최종점수 :  0.9298245614035088
# 엒뀨라씨 스코어 0.9298245614035088

# #bagging Logistic
# 최종점수 :  0.9385964912280702
# 엒뀨라씨 스코어 0.9385964912280702


#bagging DecisionTree bootstrap=False
#최종점수 :  0.9649122807017544
#엒뀨라씨 스코어 0.9649122807017544

# #Randomforest
# 최종점수 :  0.9473684210526315
# 엒뀨라씨 스코어 0.9473684210526315