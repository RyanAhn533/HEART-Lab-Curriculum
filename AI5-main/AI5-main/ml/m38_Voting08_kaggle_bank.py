import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
#import lightgbm   #pip install 필요
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor


#LogisticRegression -> 분류 모델
#삼성 기술면접에서 logisticRegression이 분류 !

path = "./_data/kaggle/Bank/"

# CSV 파일 로드
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

# 데이터 확인
print(train_csv['Geography'].value_counts())

# 데이터 변환
train_csv['Geography'] = train_csv['Geography'].replace({'France': 1, 'Spain': 2, 'Germany': 3})
test_csv['Geography'] = test_csv['Geography'].replace({'France': 1, 'Spain': 2, 'Germany': 3})
train_csv['Gender'] = train_csv['Gender'].replace({'Male': 1, 'Female': 2})
test_csv['Gender'] = test_csv['Gender'].replace({'Male': 1, 'Female': 2})


# 특정 열에 0 값을 가진 행 삭제
"""
train_csv = train_csv[train_csv['Balance'] != 0]
test_csv = test_csv[test_csv['Balance'] != 0]
"""

# 문자열 값을 가진 열 확인 및 삭제
print(train_csv.select_dtypes(include=['object']).columns)
print(test_csv.select_dtypes(include=['object']).columns)

# 'Surname' 열 삭제
train_csv = train_csv.drop(['Surname'], axis=1)
test_csv = test_csv.drop(['Surname'], axis=1)

# 데이터 저장
train_csv.to_csv(path + "replaced_train.csv")
test_csv.to_csv(path + "replaced_test.csv")

# 데이터 로드
re_train_csv = pd.read_csv(path + "replaced_train.csv", index_col=0)
re_test_csv = pd.read_csv(path + "replaced_test.csv", index_col=0)

# 데이터 확인
re_train_csv.info()
re_test_csv.info()

# 특정 열 제거
re_train_csv = re_train_csv.drop(['CustomerId'], axis=1)
re_test_csv = re_test_csv.drop(['CustomerId'], axis=1)


# 데이터 스케일링
scaler = StandardScaler()
re_train_csv_scaled = scaler.fit_transform(re_train_csv.drop(['Exited'], axis=1))
re_test_csv_scaled = scaler.transform(re_test_csv)


# 데이터프레임으로 변환
re_train_csv = pd.concat([pd.DataFrame(re_train_csv_scaled), re_train_csv['Exited'].reset_index(drop=True)], axis=1)
re_test_csv = pd.DataFrame(re_test_csv_scaled)



# 학습 데이터 분리
x = re_train_csv.drop(['Exited'], axis=1)
y = re_train_csv['Exited']
print(type(x))  # <class 'pandas.core.frame.DataFrame'>
print(type(y))  # <class 'pandas.core.series.Series'>

x = x.to_numpy()

print(re_train_csv.shape)
print(re_test_csv.shape)
print(type(x))


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1186, train_size=0.8)
