import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#LogisticRegression -> 분류 모델
#삼성 기술면접에서 logisticRegression이 분류 !

path = "./_data/dacon/diabets/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

print(train_csv.columns)
print(test_csv.columns)


x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

scaler = StandardScaler()

x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=512, train_size=0.8)
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
배깅 디찌쟌 False
최종점수 :  -0.15462937399678967
엒뀨라씨 스코어 -0.15462937399678967

배깅 디시쟌 트루
최종점수 :  0.32147397003745315
엒뀨라씨 스코어 0.32147397003745315

랜덤 뽀르느스트
최종점수 :  0.29844208132691274
엒뀨라씨 스코어 0.29844208132691274
'''