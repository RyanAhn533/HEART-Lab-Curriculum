from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


#1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)
df['target'] = datasets.target

#df.boxplot()  #population 요놈 이상해! 를 알게 되었다 
#df.plot.box()
#plt.show()

print(df.info()) #결측치 없음
print(df.describe())

#df['Population'].boxplot()  #시리즈에서 안되
#df['Population'].plt.box()  #시리즈에서 되
#df['Population'].hist(bins=50)  #시리즈에서 되
#df['target'].hist(bins=50)  #시리즈에서 되
#plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

#######################Population 로그 변환######################
x['Population'] = np.log1p(x['Population']) #지수변환 np.exp1m
#Dataframe은 series의 연속 시리즈란?
#https://kongdols-room.tistory.com/103

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=1234,)

###################### y 로그 변환 ######################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
#######################################################

#2. 모델
#model = RandomForestRegressor(random_state=1234, max_depth=5, min_samples_split=3)
model = LinearRegression()
#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
score = model.score(x_test, y_test)

print('score :', score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2_score 는?', r2)
# 로그 변환 후 r2_score 는? 0.6584197269397019
# x만 로그 변환 전 0.6584197269397019
# 전부 로그 변환 전 r2_score 는? 0.6495152533878351

#LinearRegression 로그 변환 전 r2_score 는? 0.606572212210644
#y 로그로 변환 r2_score 는? 0.606572212210644
#x만 로그로 변환 r2_score 는? 0.606598836886877
#전부 변환 r2_score 는? 0.6294707351612604