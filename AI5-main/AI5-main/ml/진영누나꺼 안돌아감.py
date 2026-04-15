
# 보스톤, 캘리포니아, 디아벳

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits

from sklearn.model_selection import train_test_split, KFold
# score 교차, val 검증
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 트레인 테스트 스플릿 이후에 케이폴드를 하겠다
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
import time


#1. 데이터
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

all = all_estimators(type_filter='classifier')


datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디저트']


all = all_estimators(type_filter='regressor')   # 모델의 개수 :  55


kfold = KFold(n_splits=5, shuffle=True, random_state=777)

best = []

start = time.time()
# i = 0 value = iris
for index, value in enumerate(datasets):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        shuffle=True,
                                                        random_state=123,
                                                        train_size=0.8,
                                                        # stratify=y
                                                        ) 
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)   # 여기서 훈련을 한 다음

    maxAccuracy = 0   # 최대의 정확도 / 최고 모델 정의
    maxName = ''      # 최대의 정확도를 가진 모델의 이름

    for name, model in all:
        try:   # try로 묶어버림
            #2. 모델
            model = model()   # model은 전부 돌아주고 xgb는 띄엄띄엄 돌아줌
            #3. 훈련
            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            acc = round(np.mean(scores), 4)
            if maxAccuracy < acc:   # acc로 정의한것. 정의 안하고 round(np.mean(scores), 4)) 써도 됨
                maxAccuracy = acc   # 최대의 정확도를 찾아서
                maxName = name      # 모델의 이름을 찾아서      

            print(name)            
 
            
            
        except:
            print(name, '은 바보 멍청이!!!')

        print("======", data_name[index], "======")
        print("======", maxName, maxAccuracy, "======")
   
end = time.time()
print('걸린시간 : ', round(end - start, 2), "초")

