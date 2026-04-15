from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping
import time as t
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, accuracy_score

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb

# 파일 경로 설정
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
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True,
                                                    random_state=123, train_size=0.8,
                                                    stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits=3

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333) #, random_state=333
#kfold 할 준비 끗

#kfold = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=333)
model = xgb.XGBClassifier()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('ACC ', scores, '\n 평균 ACC', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)

print(y_predict)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC : ', acc)
'''
France     94215
Spain      36213
Germany    34606
Name: Geography, dtype: int64
Index(['Surname'], dtype='object')
Index(['Surname'], dtype='object')
<class 'pandas.core.frame.DataFrame'>
Int64Index: 165034 entries, 0 to 165033       
Data columns (total 12 columns):
 #   Column           Non-Null Count   Dtype  
---  ------           --------------   -----  
 0   CustomerId       165034 non-null  int64  
 1   CreditScore      165034 non-null  int64
 2   Geography        165034 non-null  int64
 3   Gender           165034 non-null  int64
 4   Age              165034 non-null  float64
 5   Tenure           165034 non-null  int64
 6   Balance          165034 non-null  float64
 7   NumOfProducts    165034 non-null  int64
 8   HasCrCard        165034 non-null  float64
 9   IsActiveMember   165034 non-null  float64
 10  EstimatedSalary  165034 non-null  float64
 11  Exited           165034 non-null  int64
dtypes: float64(5), int64(7)
memory usage: 16.4 MB
<class 'pandas.core.frame.DataFrame'>
Int64Index: 110023 entries, 165034 to 275056
Data columns (total 11 columns):
 #   Column           Non-Null Count   Dtype
---  ------           --------------   -----
 0   CustomerId       110023 non-null  int64
 1   CreditScore      110023 non-null  int64
 2   Geography        110023 non-null  int64
 3   Gender           110023 non-null  int64
 4   Age              110023 non-null  float64
 5   Tenure           110023 non-null  int64
 6   Balance          110023 non-null  float64
 7   NumOfProducts    110023 non-null  int64
 8   HasCrCard        110023 non-null  float64
 9   IsActiveMember   110023 non-null  float64
 10  EstimatedSalary  110023 non-null  float64
dtypes: float64(5), int64(6)
memory usage: 10.1 MB
ACC  [0.86459588 0.86445954 0.8629144 ] 
 평균 ACC 0.864
[1 0 0 ... 0 1 0]
110916    1
81797     0
155703    0
44621     0
11314     0
         ..
160424    0
23945     0
27297     0
67177     1
120419    0
Name: Exited, Length: 33007, dtype: int64
cross_val_predict ACC :  0.8567273608628473
'''