import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier

path = 'C:\\ai5\\_data\\dacon\\생명연구\\'
train_csv = pd.read_csv(path + "train.csv")
test_csv = pd.read_csv(path + "test.csv")

print(train_csv, train_csv.shape)         # (6201, 4386)
print(test_csv, test_csv.shape)           # (2546, 4385) 

########## LabelEncoding ##########
le = LabelEncoder()
y = le.fit_transform(train_csv['SUBCLASS'])

# 변환된 레이블 확인
aaa = {}
for i, label in enumerate(le.classes_):
    aaa[label] = i
# print(aaa)
# 26개, {'ACC': 0, 'BLCA': 1, 'BRCA': 2, 'CESC': 3, 'COAD': 4, 'DLBC': 5, 'GBMLGG': 6, 'HNSC': 7, 'KIPAN': 8, 'KIRC': 9, 'LAML': 10, 'LGG': 11, 'LIHC': 12, 'LUAD': 13, 'LUSC': 14, 'OV': 15, 'PAAD': 16, 'PCPG': 17, 'PRAD': 18, 'SARC': 19, 'SKCM': 20, 'STES': 21, 'TGCT': 22, 'THCA': 23, 'THYM': 24, 'UCEC': 25}

x = train_csv.drop(columns=['SUBCLASS', 'ID'])
test_csv = test_csv.drop(columns=['ID'])
# print(x.shape, y.shape, test_csv.shape)   # (6201, 4384) (6201,) (2546, 4384)

categorical_columns = x.select_dtypes(include=['object', 'category']).columns
# print(categorical_columns)      # 컬럼명 추출
# Index(['A2M', 'AAAS', 'AADAT', ... 'ZW10', 'ZWINT', 'ZYX'], dtype='object', length=4384)
# print(len(categorical_columns)) # 4384

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y, random_state=123)

parameters = {
     'learning_rate' : [0.01, 0.03, 0.05, 0.1, 0.2],
     'depth' : [4, 6, 8, 10, 12],
     'l2_leaf_reg' : [1, 3, 5, 7, 10],
     'bagging_temperature' : [0.0, 0.5, 1.0, 2.0, 5.0],
     'border_count' : [32, 64, 128, 255],
     'random_strength' : [1, 5, 10],
}

cat_features = list(range(x_train.shape[1]))
# print(cat_features)


#2. 모델 구성
model = CatBoostClassifier(
    iterations=2000,            # 트리 개수 (Default: 500)
    learning_rate=0.03,         # 학습률 (Default: 0.03)
    depth=6,                    # 트리 깊이 (Default: 6)
    l2_leaf_reg=5,              # L2 정규화 (Default: 3)
    bagging_temperature=1.0,    # 배깅 온도 (Default: 1.0)
    random_strength=5,          # 랜덤성 추가 (Default: 1)
    border_count=128,           # 연속형 변수 처리 (Default: 254)
    task_type="GPU",            # GPU 사용 (Default: 'CPU')
    devices='0',                # 첫 번째 GPU 사용 (Default: 모든 GPU 사용)
    early_stopping_rounds=100,  # 조기 종료 (Default: None)
    verbose=10,                 # 매 10번째 반복마다 출력 (Default: 100)
    # eval_metric='F1',         # F1-score를 평가지표로 설정 (macro나 micro같은 다중분류는 없다
    cat_features=cat_features
    )

#3. 훈련
start = time.time()
model.fit(x_train, y_train, 
          eval_set=[(x_test, y_test)], 
          verbose=False)
end = time.time()

joblib.dump(model, './_save/dacon02/d02_16.pkl')

#4. 평가, 예측
print('model.score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
f1 = f1_score(y_test, y_predict, average = 'macro')

result = model.predict(x_test)
result = le.inverse_transform(result)

submisson = pd.read_csv(path + "sample_submission.csv")
submisson["SUBCLASS"] = result
submisson.to_csv(path + 'baseline_submission_16.csv', encoding='UTF-8-sig', index=False)

print('f1_score :', f1, '/ time :', round(end-start, 2), '초 >')

'''
90
ValueError: Length of values (1241) does not match length of index (2546)

'''
