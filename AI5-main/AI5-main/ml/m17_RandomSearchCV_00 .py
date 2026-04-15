import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import pandas as pd


#1.데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, 
                                                    random_state=333,
                                                    stratify=y, train_size=0.8)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parametes = [{"C":[1, 10, 100, 100], 'kernel':['linear', 'sigmoid'],
              'degree':[3,4,5]}, #24
             {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]}, #6,
             {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.01, 0.001, 0.0001], 'degree':[3,4]} #24
             ] #총 54
 
model = RandomizedSearchCV(SVC(), parametes, cv=kfold, verbose=1,
                     refit=True,
                     n_jobs=-1, #cpu코어 n개중 n개 쓰겟다 -1 이니전부 다 쓰겠다!,
                     n_iter=9, #반복횟수
                     random_state=3333,
                     )
#gridsearch 랩핑하기!

start_time = time.time()
model.fit(x_train, y_train)

end_time = time.time()

print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 : ', model.best_params_)
#최적의 매개변수 : SVC(C=10, kernel='linear')
#최적의 파라미터 :  {'C': 10, 'degree': 3, 'kernel': 'linear'}


print('best_score : ', model.best_score_)
#훈련에서의 best score -> train데이터에서의 스코어
#best_score :  0.9916666666666668

print('model.score : ', model.score(x_test, y_test))
#model.score :  0.9
#얘가 실질적 스코어 - test와의 스코어

y_predict = model.predict(x_test)
print('accuracy_score', accuracy_score(y_test, y_predict))
#accuracy_score 0.9
# best_score = acc 

y_pred_best = model.best_estimator_.predict(x_test) # 요새끼가 최고다 얘를 써라!
print('최적 튠 ACC :', accuracy_score(y_test, y_pred_best))

#최적 튠 ACC : 0.9

print('걸린시간 : ', round(end_time-start_time, 2), '초')
#걸린시간 :  1.43 초
import pandas as pd
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))
print(pd.DataFrame(model.cv_results_).columns)
'''
Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
       'param_C', 'param_degree', 'param_kernel', 'param_gamma', 'params',
       'split0_test_score', 'split1_test_score', 'split2_test_score',
       'split3_test_score', 'split4_test_score', 'mean_test_score',
       'std_test_score', 'rank_test_score'],
      dtype='object')
'''

path = 'C:\\프로그램\\ai5\\_data\\_save\\m15_GS_CV_01\\'
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)\
    .to_csv(path + 'm17_randomizer.csv'))