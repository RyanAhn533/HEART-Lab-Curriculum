import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
import time
warnings.filterwarnings('ignore')

#cross val score - 교차검증점수
# 5개로 짜른 데이터들마다 교차 검증 점수를 매긴다

#1. 데이터
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

all = all_estimators(type_filter='classifier')


datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디저트']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=787)

start_time = time.time()

for index, value in enumerate(datasets):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8,
    stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    for name, model in all:
        try:
            #2. 모델
            model = model()
            #3. 훈련
            scores = cross_val_score(model, x_train, y_train , cv= kfold)
            print('=========================',data_name[index],name,'========================')
            print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))
        
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = accuracy_score(y_test,y_predict)
            print('cross_val_predict ACC :', acc)
        
            #4. 평가
            acc = model.score(x_test,y_test)
            print(name, '의 정답률 :', acc)
        except:
            print(name,'은 바보 다스베이더')

end_time = time.time()
print('걸린 시간 :', round(end_time - start_time, 2), '초')
'''
========================= 아이리스 AdaBoostClassifier ========================
ACC :  [0.95833333 0.95833333 0.95833333 0.95833333 0.875     ] 
 평균 ACC :  0.9417
cross_val_predict ACC : 0.9333333333333333
AdaBoostClassifier 은 바보 다스베이더
========================= 아이리스 BaggingClassifier ========================
ACC :  [0.95833333 1.         0.95833333 1.         0.875     ] 
 평균 ACC :  0.9583
cross_val_predict ACC : 0.9666666666666667
BaggingClassifier 은 바보 다스베이더
========================= 아이리스 BernoulliNB ========================
ACC :  [0.83333333 0.83333333 0.83333333 0.70833333 0.66666667]        
 평균 ACC :  0.775
cross_val_predict ACC : 0.6
BernoulliNB 은 바보 다스베이더
========================= 아이리스 CalibratedClassifierCV ========================
ACC :  [0.95833333 0.95833333 0.875      0.875      0.79166667] 
 평균 ACC :  0.8917
cross_val_predict ACC : 0.8333333333333334
CalibratedClassifierCV 은 바보 다스베이더
CategoricalNB 은 바보 다스베이더
ClassifierChain 은 바보 다스베이더
ComplementNB 은 바보 다스베이더
========================= 아이리스 DecisionTreeClassifier ========================
ACC :  [0.95833333 1.         0.91666667 0.95833333 0.875     ]
 평균 ACC :  0.9417
cross_val_predict ACC : 0.9666666666666667
DecisionTreeClassifier 은 바보 다스베이더
========================= 아이리스 DummyClassifier ========================
ACC :  [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333]
 평균 ACC :  0.3333
cross_val_predict ACC : 0.3333333333333333
DummyClassifier 은 바보 다스베이더
========================= 아이리스 ExtraTreeClassifier ========================
ACC :  [0.95833333 1.         0.91666667 1.         0.91666667]
 평균 ACC :  0.9583
cross_val_predict ACC : 0.9666666666666667
ExtraTreeClassifier 은 바보 다스베이더
========================= 아이리스 ExtraTreesClassifier ========================
ACC :  [0.95833333 1.         0.95833333 1.         0.91666667] 
 평균 ACC :  0.9667
cross_val_predict ACC : 0.9666666666666667
ExtraTreesClassifier 은 바보 다스베이더
FixedThresholdClassifier 은 바보 다스베이더
========================= 아이리스 GaussianNB ========================
ACC :  [0.95833333 1.         0.95833333 0.95833333 0.875     ]
 평균 ACC :  0.95
cross_val_predict ACC : 0.9666666666666667
GaussianNB 은 바보 다스베이더
========================= 아이리스 GaussianProcessClassifier ========================
ACC :  [0.95833333 1.         0.95833333 1.         0.875     ] 
 평균 ACC :  0.9583
cross_val_predict ACC : 0.9666666666666667
GaussianProcessClassifier 은 바보 다스베이더
========================= 아이리스 GradientBoostingClassifier ========================
ACC :  [0.95833333 1.         0.95833333 0.95833333 0.875     ] 
 평균 ACC :  0.95
cross_val_predict ACC : 0.9666666666666667
GradientBoostingClassifier 은 바보 다스베이더
========================= 아이리스 HistGradientBoostingClassifier ========================
ACC :  [0.95833333 1.         0.91666667 1.         0.875     ] 
 평균 ACC :  0.95
cross_val_predict ACC : 0.3333333333333333
HistGradientBoostingClassifier 은 바보 다스베이더
========================= 아이리스 KNeighborsClassifier ========================
ACC :  [0.95833333 1.         0.95833333 1.         0.95833333]
 평균 ACC :  0.975
cross_val_predict ACC : 0.9
KNeighborsClassifier 은 바보 다스베이더
========================= 아이리스 LabelPropagation ========================
ACC :  [0.91666667 1.         0.91666667 1.         0.875     ]
 평균 ACC :  0.9417
cross_val_predict ACC : 0.8333333333333334
LabelPropagation 은 바보 다스베이더
========================= 아이리스 LabelSpreading ========================
ACC :  [0.91666667 1.         0.91666667 1.         0.875     ]
 평균 ACC :  0.9417
cross_val_predict ACC : 0.8333333333333334
LabelSpreading 은 바보 다스베이더
========================= 아이리스 LinearDiscriminantAnalysis ========================
ACC :  [1.    1.    1.    1.    0.875]
 평균 ACC :  0.975
cross_val_predict ACC : 1.0
LinearDiscriminantAnalysis 은 바보 다스베이더
========================= 아이리스 LinearSVC ========================
ACC :  [0.95833333 1.         1.         1.         0.79166667]
 평균 ACC :  0.95
cross_val_predict ACC : 0.9
LinearSVC 은 바보 다스베이더
========================= 아이리스 LogisticRegression ========================
ACC :  [1.         1.         0.95833333 1.         0.875     ]
 평균 ACC :  0.9667
cross_val_predict ACC : 0.9
LogisticRegression 은 바보 다스베이더
========================= 아이리스 LogisticRegressionCV ========================
ACC :  [1.         1.         0.95833333 1.         0.91666667] 
 평균 ACC :  0.975
cross_val_predict ACC : 0.9333333333333333
LogisticRegressionCV 은 바보 다스베이더
========================= 아이리스 MLPClassifier ========================
ACC :  [0.95833333 1.         0.95833333 1.         0.83333333] 
 평균 ACC :  0.95
cross_val_predict ACC : 0.9
MLPClassifier 은 바보 다스베이더
MultiOutputClassifier 은 바보 다스베이더
MultinomialNB 은 바보 다스베이더
========================= 아이리스 NearestCentroid ========================
ACC :  [0.83333333 0.95833333 0.95833333 0.83333333 0.83333333]
 평균 ACC :  0.8833
cross_val_predict ACC : 0.7666666666666667
NearestCentroid 은 바보 다스베이더
========================= 아이리스 NuSVC ========================
ACC :  [0.95833333 1.         0.95833333 1.         0.91666667]
 평균 ACC :  0.9667
cross_val_predict ACC : 0.9666666666666667
NuSVC 은 바보 다스베이더
OneVsOneClassifier 은 바보 다스베이더
OneVsRestClassifier 은 바보 다스베이더
OutputCodeClassifier 은 바보 다스베이더
========================= 아이리스 PassiveAggressiveClassifier ========================
ACC :  [0.91666667 0.95833333 0.95833333 0.91666667 0.83333333]
 평균 ACC :  0.9167
cross_val_predict ACC : 0.8666666666666667
PassiveAggressiveClassifier 은 바보 다스베이더
========================= 아이리스 Perceptron ========================
ACC :  [0.95833333 1.         0.83333333 0.91666667 0.66666667]
 평균 ACC :  0.875
cross_val_predict ACC : 0.8
Perceptron 은 바보 다스베이더
========================= 아이리스 QuadraticDiscriminantAnalysis ========================
ACC :  [1.         1.         1.         1.         0.91666667]
 평균 ACC :  0.9833
cross_val_predict ACC : 0.8333333333333334
QuadraticDiscriminantAnalysis 은 바보 다스베이더
========================= 아이리스 RadiusNeighborsClassifier ========================
ACC :  [0.95833333        nan 0.95833333        nan        nan]
 평균 ACC :  nan
RadiusNeighborsClassifier 은 바보 다스베이더
========================= 아이리스 RandomForestClassifier ========================
ACC :  [0.95833333 1.         0.95833333 0.95833333 0.875     ] 
 평균 ACC :  0.95
cross_val_predict ACC : 0.9666666666666667
RandomForestClassifier 은 바보 다스베이더
========================= 아이리스 RidgeClassifier ========================
ACC :  [0.83333333 0.95833333 0.83333333 0.79166667 0.79166667]
 평균 ACC :  0.8417
cross_val_predict ACC : 0.8333333333333334
RidgeClassifier 은 바보 다스베이더
========================= 아이리스 RidgeClassifierCV ========================
ACC :  [0.83333333 0.95833333 0.83333333 0.79166667 0.79166667]
 평균 ACC :  0.8417
cross_val_predict ACC : 0.8333333333333334
RidgeClassifierCV 은 바보 다스베이더
========================= 아이리스 SGDClassifier ========================
ACC :  [0.95833333 1.         0.91666667 0.91666667 0.83333333]
 평균 ACC :  0.925
cross_val_predict ACC : 0.8
SGDClassifier 은 바보 다스베이더
========================= 아이리스 SVC ========================
ACC :  [0.95833333 1.         0.95833333 1.         0.91666667]
 평균 ACC :  0.9667
cross_val_predict ACC : 0.8333333333333334
SVC 은 바보 다스베이더
StackingClassifier 은 바보 다스베이더
TunedThresholdClassifierCV 은 바보 다스베이더
VotingClassifier 은 바보 다스베이더
========================= 캔서 AdaBoostClassifier ========================
ACC :  [0.96703297 0.93406593 0.94505495 0.98901099 0.94505495] 
 평균 ACC :  0.956
cross_val_predict ACC : 0.9298245614035088
AdaBoostClassifier 은 바보 다스베이더
========================= 캔서 BaggingClassifier ========================
ACC :  [0.96703297 0.94505495 0.93406593 0.96703297 0.95604396] 
 평균 ACC :  0.9538
cross_val_predict ACC : 0.9473684210526315
BaggingClassifier 은 바보 다스베이더
========================= 캔서 BernoulliNB ========================
ACC :  [0.94505495 0.93406593 0.94505495 0.95604396 0.9010989 ]
 평균 ACC :  0.9363
cross_val_predict ACC : 0.9473684210526315
BernoulliNB 은 바보 다스베이더
========================= 캔서 CalibratedClassifierCV ========================
ACC :  [0.97802198 0.93406593 0.97802198 0.98901099 0.94505495] 
 평균 ACC :  0.9648
cross_val_predict ACC : 0.9385964912280702
CalibratedClassifierCV 은 바보 다스베이더
CategoricalNB 은 바보 다스베이더
ClassifierChain 은 바보 다스베이더
ComplementNB 은 바보 다스베이더
========================= 캔서 DecisionTreeClassifier ========================
ACC :  [0.95604396 0.9010989  0.93406593 0.96703297 0.91208791]
 평균 ACC :  0.9341
cross_val_predict ACC : 0.9473684210526315
DecisionTreeClassifier 은 바보 다스베이더
========================= 캔서 DummyClassifier ========================
ACC :  [0.62637363 0.62637363 0.62637363 0.62637363 0.62637363]
 평균 ACC :  0.6264
cross_val_predict ACC : 0.631578947368421
DummyClassifier 은 바보 다스베이더
========================= 캔서 ExtraTreeClassifier ========================
ACC :  [0.87912088 0.91208791 0.87912088 0.87912088 0.91208791]
 평균 ACC :  0.8923
cross_val_predict ACC : 0.9210526315789473
ExtraTreeClassifier 은 바보 다스베이더
========================= 캔서 ExtraTreesClassifier ========================
ACC :  [0.97802198 0.95604396 0.93406593 0.98901099 0.96703297] 
 평균 ACC :  0.9648
cross_val_predict ACC : 0.9473684210526315
ExtraTreesClassifier 은 바보 다스베이더
FixedThresholdClassifier 은 바보 다스베이더
========================= 캔서 GaussianNB ========================
ACC :  [0.93406593 0.94505495 0.92307692 0.96703297 0.93406593]
 평균 ACC :  0.9407
cross_val_predict ACC : 0.9385964912280702
GaussianNB 은 바보 다스베이더
========================= 캔서 GaussianProcessClassifier ========================
ACC :  [0.97802198 0.94505495 0.96703297 0.98901099 0.95604396] 
 평균 ACC :  0.967
cross_val_predict ACC : 0.9649122807017544
GaussianProcessClassifier 은 바보 다스베이더
========================= 캔서 GradientBoostingClassifier ========================
ACC :  [0.97802198 0.93406593 0.93406593 0.97802198 0.94505495] 
 평균 ACC :  0.9538
cross_val_predict ACC : 0.9473684210526315
GradientBoostingClassifier 은 바보 다스베이더
========================= 캔서 HistGradientBoostingClassifier ========================
ACC :  [0.97802198 0.94505495 0.93406593 1.         0.95604396] 
 평균 ACC :  0.9626
cross_val_predict ACC : 0.9473684210526315
HistGradientBoostingClassifier 은 바보 다스베이더
========================= 캔서 KNeighborsClassifier ========================
ACC :  [0.95604396 0.94505495 0.95604396 0.98901099 0.95604396] 
 평균 ACC :  0.9604
cross_val_predict ACC : 0.956140350877193
KNeighborsClassifier 은 바보 다스베이더
========================= 캔서 LabelPropagation ========================
ACC :  [0.96703297 0.93406593 0.95604396 0.96703297 0.96703297] 
 평균 ACC :  0.9582
cross_val_predict ACC : 0.956140350877193
LabelPropagation 은 바보 다스베이더
========================= 캔서 LabelSpreading ========================
ACC :  [0.96703297 0.93406593 0.95604396 0.96703297 0.96703297] 
 평균 ACC :  0.9582
cross_val_predict ACC : 0.956140350877193
LabelSpreading 은 바보 다스베이더
========================= 캔서 LinearDiscriminantAnalysis ========================
ACC :  [0.98901099 0.94505495 0.93406593 0.97802198 0.93406593]
 평균 ACC :  0.956
cross_val_predict ACC : 0.9298245614035088
LinearDiscriminantAnalysis 은 바보 다스베이더
========================= 캔서 LinearSVC ========================
ACC :  [0.98901099 0.95604396 0.98901099 0.98901099 0.97802198]
 평균 ACC :  0.9802
cross_val_predict ACC : 0.9298245614035088
LinearSVC 은 바보 다스베이더
========================= 캔서 LogisticRegression ========================
ACC :  [1.         0.98901099 0.98901099 0.98901099 0.96703297] 
 평균 ACC :  0.9868
cross_val_predict ACC : 0.9473684210526315
LogisticRegression 은 바보 다스베이더
========================= 캔서 LogisticRegressionCV ========================
ACC :  [1.         0.97802198 0.98901099 0.98901099 0.96703297] 
 평균 ACC :  0.9846
cross_val_predict ACC : 0.956140350877193
LogisticRegressionCV 은 바보 다스베이더
========================= 캔서 MLPClassifier ========================
ACC :  [0.98901099 0.95604396 0.96703297 0.98901099 0.95604396] 
 평균 ACC :  0.9714
cross_val_predict ACC : 0.9473684210526315
MLPClassifier 은 바보 다스베이더
MultiOutputClassifier 은 바보 다스베이더
MultinomialNB 은 바보 다스베이더
========================= 캔서 NearestCentroid ========================
ACC :  [0.92307692 0.93406593 0.89010989 0.95604396 0.91208791]
 평균 ACC :  0.9231
cross_val_predict ACC : 0.9473684210526315
NearestCentroid 은 바보 다스베이더
========================= 캔서 NuSVC ========================
ACC :  [0.95604396 0.93406593 0.92307692 0.96703297 0.92307692]
 평균 ACC :  0.9407
cross_val_predict ACC : 0.956140350877193
NuSVC 은 바보 다스베이더
OneVsOneClassifier 은 바보 다스베이더
OneVsRestClassifier 은 바보 다스베이더
OutputCodeClassifier 은 바보 다스베이더
========================= 캔서 PassiveAggressiveClassifier ========================
ACC :  [0.95604396 0.97802198 0.97802198 0.98901099 0.95604396]
 평균 ACC :  0.9714
cross_val_predict ACC : 0.9298245614035088
PassiveAggressiveClassifier 은 바보 다스베이더
========================= 캔서 Perceptron ========================
ACC :  [0.96703297 0.96703297 0.96703297 0.98901099 0.96703297]
 평균 ACC :  0.9714
cross_val_predict ACC : 0.9385964912280702
Perceptron 은 바보 다스베이더
========================= 캔서 QuadraticDiscriminantAnalysis ========================
ACC :  [0.95604396 0.96703297 0.95604396 0.97802198 0.92307692]
 평균 ACC :  0.956
cross_val_predict ACC : 0.8596491228070176
QuadraticDiscriminantAnalysis 은 바보 다스베이더
========================= 캔서 RadiusNeighborsClassifier ========================
ACC :  [nan nan nan nan nan]
 평균 ACC :  nan
RadiusNeighborsClassifier 은 바보 다스베이더
========================= 캔서 RandomForestClassifier ========================
ACC :  [0.96703297 0.93406593 0.93406593 0.98901099 0.97802198] 
 평균 ACC :  0.9604
cross_val_predict ACC : 0.9385964912280702
RandomForestClassifier 은 바보 다스베이더
========================= 캔서 RidgeClassifier ========================
ACC :  [0.98901099 0.93406593 0.93406593 0.97802198 0.95604396]
 평균 ACC :  0.9582
cross_val_predict ACC : 0.9473684210526315
RidgeClassifier 은 바보 다스베이더
========================= 캔서 RidgeClassifierCV ========================
ACC :  [0.97802198 0.93406593 0.93406593 0.97802198 0.95604396]
 평균 ACC :  0.956
cross_val_predict ACC : 0.9649122807017544
RidgeClassifierCV 은 바보 다스베이더
========================= 캔서 SGDClassifier ========================
ACC :  [0.95604396 0.95604396 0.98901099 0.97802198 0.93406593]
 평균 ACC :  0.9626
cross_val_predict ACC : 0.9385964912280702
SGDClassifier 은 바보 다스베이더
========================= 캔서 SVC ========================
ACC :  [0.98901099 0.97802198 0.97802198 1.         0.95604396]
 평균 ACC :  0.9802
cross_val_predict ACC : 0.956140350877193
SVC 은 바보 다스베이더
StackingClassifier 은 바보 다스베이더
TunedThresholdClassifierCV 은 바보 다스베이더
VotingClassifier 은 바보 다스베이더
========================= 와인 AdaBoostClassifier ========================
ACC :  [0.86206897 0.79310345 0.89285714 1.         0.96428571] 
 평균 ACC :  0.9025
cross_val_predict ACC : 0.8611111111111112
AdaBoostClassifier 은 바보 다스베이더
========================= 와인 BaggingClassifier ========================
ACC :  [0.86206897 0.93103448 0.92857143 0.96428571 0.96428571] 
 평균 ACC :  0.93
cross_val_predict ACC : 0.9166666666666666
BaggingClassifier 은 바보 다스베이더
========================= 와인 BernoulliNB ========================
ACC :  [0.89655172 0.89655172 0.92857143 0.96428571 1.        ]
 평균 ACC :  0.9372
cross_val_predict ACC : 0.9444444444444444
BernoulliNB 은 바보 다스베이더
========================= 와인 CalibratedClassifierCV ========================
ACC :  [0.96551724 0.93103448 0.96428571 1.         1.        ] 
 평균 ACC :  0.9722
cross_val_predict ACC : 0.9722222222222222
CalibratedClassifierCV 은 바보 다스베이더
CategoricalNB 은 바보 다스베이더
ClassifierChain 은 바보 다스베이더
ComplementNB 은 바보 다스베이더
========================= 와인 DecisionTreeClassifier ========================
ACC :  [0.86206897 0.93103448 0.92857143 0.92857143 0.96428571]
 평균 ACC :  0.9229
cross_val_predict ACC : 0.8888888888888888
DecisionTreeClassifier 은 바보 다스베이더
========================= 와인 DummyClassifier ========================
ACC :  [0.4137931  0.4137931  0.39285714 0.39285714 0.39285714]
 평균 ACC :  0.4012
cross_val_predict ACC : 0.3888888888888889
DummyClassifier 은 바보 다스베이더
========================= 와인 ExtraTreeClassifier ========================
ACC :  [0.86206897 0.68965517 0.96428571 0.85714286 0.92857143]
 평균 ACC :  0.8603
cross_val_predict ACC : 0.8055555555555556
ExtraTreeClassifier 은 바보 다스베이더
========================= 와인 ExtraTreesClassifier ========================
ACC :  [1.         0.96551724 1.         1.         1.        ] 
 평균 ACC :  0.9931
cross_val_predict ACC : 0.9166666666666666
ExtraTreesClassifier 은 바보 다스베이더
FixedThresholdClassifier 은 바보 다스베이더
========================= 와인 GaussianNB ========================
ACC :  [0.93103448 0.93103448 1.         1.         1.        ]
 평균 ACC :  0.9724
cross_val_predict ACC : 0.9722222222222222
GaussianNB 은 바보 다스베이더
========================= 와인 GaussianProcessClassifier ========================
ACC :  [0.93103448 0.96551724 1.         0.96428571 0.96428571] 
 평균 ACC :  0.965
cross_val_predict ACC : 0.9444444444444444
GaussianProcessClassifier 은 바보 다스베이더
========================= 와인 GradientBoostingClassifier ========================
ACC :  [0.86206897 0.82758621 0.92857143 0.92857143 0.96428571] 
 평균 ACC :  0.9022
cross_val_predict ACC : 0.9444444444444444
GradientBoostingClassifier 은 바보 다스베이더
========================= 와인 HistGradientBoostingClassifier ========================
ACC :  [0.89655172 0.93103448 1.         1.         0.96428571] 
 평균 ACC :  0.9584
cross_val_predict ACC : 0.3888888888888889
HistGradientBoostingClassifier 은 바보 다스베이더
========================= 와인 KNeighborsClassifier ========================
ACC :  [0.96551724 0.96551724 0.92857143 0.96428571 1.        ]
 평균 ACC :  0.9648
cross_val_predict ACC : 0.9444444444444444
KNeighborsClassifier 은 바보 다스베이더
========================= 와인 LabelPropagation ========================
ACC :  [0.93103448 0.93103448 1.         0.96428571 0.96428571]
 평균 ACC :  0.9581
cross_val_predict ACC : 0.8611111111111112
LabelPropagation 은 바보 다스베이더
========================= 와인 LabelSpreading ========================
ACC :  [0.93103448 0.93103448 1.         0.96428571 0.96428571]
 평균 ACC :  0.9581
cross_val_predict ACC : 0.8611111111111112
LabelSpreading 은 바보 다스베이더
========================= 와인 LinearDiscriminantAnalysis ========================
ACC :  [0.93103448 0.96551724 1.         1.         1.        ]
 평균 ACC :  0.9793
cross_val_predict ACC : 0.9722222222222222
LinearDiscriminantAnalysis 은 바보 다스베이더
========================= 와인 LinearSVC ========================
ACC :  [0.93103448 0.96551724 0.96428571 1.         1.        ]
 평균 ACC :  0.9722
cross_val_predict ACC : 0.9722222222222222
LinearSVC 은 바보 다스베이더
========================= 와인 LogisticRegression ========================
ACC :  [0.93103448 0.93103448 0.92857143 1.         1.        ]
 평균 ACC :  0.9581
cross_val_predict ACC : 0.9444444444444444
LogisticRegression 은 바보 다스베이더
========================= 와인 LogisticRegressionCV ========================
ACC :  [0.86206897 0.93103448 0.92857143 1.         1.        ] 
 평균 ACC :  0.9443
cross_val_predict ACC : 0.9444444444444444
LogisticRegressionCV 은 바보 다스베이더
========================= 와인 MLPClassifier ========================
ACC :  [0.96551724 0.96551724 0.96428571 1.         1.        ] 
 평균 ACC :  0.9791
cross_val_predict ACC : 0.9722222222222222
MLPClassifier 은 바보 다스베이더
MultiOutputClassifier 은 바보 다스베이더
MultinomialNB 은 바보 다스베이더
========================= 와인 NearestCentroid ========================
ACC :  [0.93103448 0.96551724 0.96428571 1.         0.96428571]
 평균 ACC :  0.965
cross_val_predict ACC : 0.9444444444444444
NearestCentroid 은 바보 다스베이더
========================= 와인 NuSVC ========================
ACC :  [0.93103448 0.89655172 1.         1.         1.        ]
 평균 ACC :  0.9655
cross_val_predict ACC : 1.0
NuSVC 은 바보 다스베이더
OneVsOneClassifier 은 바보 다스베이더
OneVsRestClassifier 은 바보 다스베이더
OutputCodeClassifier 은 바보 다스베이더
========================= 와인 PassiveAggressiveClassifier ========================
ACC :  [0.93103448 0.96551724 0.96428571 1.         1.        ]
 평균 ACC :  0.9722
cross_val_predict ACC : 0.9722222222222222
PassiveAggressiveClassifier 은 바보 다스베이더
========================= 와인 Perceptron ========================
ACC :  [0.96551724 0.93103448 0.92857143 1.         1.        ]
 평균 ACC :  0.965
cross_val_predict ACC : 0.9722222222222222
Perceptron 은 바보 다스베이더
========================= 와인 QuadraticDiscriminantAnalysis ========================
ACC :  [0.96551724 0.96551724 1.         0.96428571 1.        ]
 평균 ACC :  0.9791
cross_val_predict ACC : 0.6666666666666666
QuadraticDiscriminantAnalysis 은 바보 다스베이더
========================= 와인 RadiusNeighborsClassifier ========================
ACC :  [nan nan nan nan nan]
 평균 ACC :  nan
RadiusNeighborsClassifier 은 바보 다스베이더
========================= 와인 RandomForestClassifier ========================
ACC :  [0.93103448 0.96551724 1.         1.         1.        ] 
 평균 ACC :  0.9793
cross_val_predict ACC : 0.9444444444444444
RandomForestClassifier 은 바보 다스베이더
========================= 와인 RidgeClassifier ========================
ACC :  [1.         0.96551724 0.96428571 1.         1.        ]
 평균 ACC :  0.986
cross_val_predict ACC : 0.9722222222222222
RidgeClassifier 은 바보 다스베이더
========================= 와인 RidgeClassifierCV ========================
ACC :  [1.         0.96551724 0.96428571 1.         1.        ]
 평균 ACC :  0.986
cross_val_predict ACC : 0.9444444444444444
RidgeClassifierCV 은 바보 다스베이더
========================= 와인 SGDClassifier ========================
ACC :  [0.93103448 0.93103448 0.96428571 1.         1.        ]
 평균 ACC :  0.9653
cross_val_predict ACC : 0.9444444444444444
SGDClassifier 은 바보 다스베이더
========================= 와인 SVC ========================
ACC :  [0.96551724 0.89655172 1.         1.         1.        ]
 평균 ACC :  0.9724
cross_val_predict ACC : 1.0
SVC 은 바보 다스베이더
StackingClassifier 은 바보 다스베이더
TunedThresholdClassifierCV 은 바보 다스베이더
VotingClassifier 은 바보 다스베이더
========================= 디저트 AdaBoostClassifier ========================
ACC :  [0.34375    0.26736111 0.25783972 0.32055749 0.28222997] 
 평균 ACC :  0.2943
cross_val_predict ACC : 0.28055555555555556
AdaBoostClassifier 은 바보 다스베이더
========================= 디저트 BaggingClassifier ========================
ACC :  [0.92013889 0.90277778 0.94076655 0.93728223 0.95121951] 
 평균 ACC :  0.9304
cross_val_predict ACC : 0.8666666666666667
BaggingClassifier 은 바보 다스베이더
========================= 디저트 BernoulliNB ========================
ACC :  [0.87152778 0.88888889 0.87456446 0.87456446 0.91637631]
 평균 ACC :  0.8852
cross_val_predict ACC : 0.85
BernoulliNB 은 바보 다스베이더
========================= 디저트 CalibratedClassifierCV ========================
ACC :  [0.96527778 0.95138889 0.95121951 0.95818815 0.94076655] 
 평균 ACC :  0.9534
cross_val_predict ACC : 0.9055555555555556
CalibratedClassifierCV 은 바보 다스베이더
CategoricalNB 은 바보 다스베이더
ClassifierChain 은 바보 다스베이더
ComplementNB 은 바보 다스베이더
========================= 디저트 DecisionTreeClassifier ========================
ACC :  [0.80208333 0.8125     0.81533101 0.83275261 0.84320557] 
 평균 ACC :  0.8212
cross_val_predict ACC : 0.7666666666666667
DecisionTreeClassifier 은 바보 다스베이더
========================= 디저트 DummyClassifier ========================
ACC :  [0.10069444 0.10069444 0.1010453  0.1010453  0.1010453 ]
 평균 ACC :  0.1009
cross_val_predict ACC : 0.09722222222222222
DummyClassifier 은 바보 다스베이더
========================= 디저트 ExtraTreeClassifier ========================
ACC :  [0.75       0.81944444 0.7804878  0.78397213 0.7630662 ]
 평균 ACC :  0.7794
cross_val_predict ACC : 0.6861111111111111
ExtraTreeClassifier 은 바보 다스베이더
========================= 디저트 ExtraTreesClassifier ========================
ACC :  [0.98263889 0.96875    0.98606272 0.97560976 0.97909408] 
 평균 ACC :  0.9784
cross_val_predict ACC : 0.9583333333333334
ExtraTreesClassifier 은 바보 다스베이더
FixedThresholdClassifier 은 바보 다스베이더
========================= 디저트 GaussianNB ========================
ACC :  [0.78125    0.74305556 0.8466899  0.79094077 0.82229965]
 평균 ACC :  0.7968
cross_val_predict ACC : 0.8
GaussianNB 은 바보 다스베이더
'''