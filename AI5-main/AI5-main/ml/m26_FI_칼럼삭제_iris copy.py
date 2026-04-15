# 랜덤뽀레스트 쓰라
# 판다스로 바꿔서 컬럼 삭제
#pd.DataFrame
#컬럼명 ㅣ datasets.feature_names 안에 있지
#피처 임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거
#데이터셋 재구성한 후
#기존 모델결과 비교!!!!!!!
# .drop해서 날리날리
# 기존 랜덤 포레스트 결과보다 향상 시켜라
#각각의 피처가 성능에 얼마나 기여했는지 명시 할 수 있다.
# 트리 구조의 모델들은 애초에 가지고 있음
# xgboost , dicision tree,  등 등

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

random_state = 1223

# 1. 데이터 로드
dataset = load_iris()
print(dataset.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


# x, y 설정
x = dataset.data  # df.data 대신 df.values 사용
y = dataset.target
from sklearn.model_selection import train_test_split
#나라시 = flatten

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    shuffle=True,
                                                    stratify=y,
                                                    random_state=random_state, train_size=0.8
                                                    )

#2. 모델구성

model1 = DecisionTreeClassifier(random_state=random_state)


models = [model1]
print("random stae 는? ", random_state)
for model in models:
    model.fit(x_train, y_train)
    print("++++++++++++++++++++++++++++++", model.__class__.__name__, "+++++++++++++++++++++++")
    print('acc', model.score(x_test, y_test))
    print(model.feature_importances_)

    # DataFrame 생성 및 'sepal width (cm)' 열 제거
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    for i in enumerate (model.feature_importances_):
        np.percentile(model.feature_importances_, [10, 20, 30, 40, 50])
        