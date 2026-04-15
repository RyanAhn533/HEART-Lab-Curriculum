import numpy as np
import pandas as pd
import time
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import random as rn
import tensorflow as tf
tf.random.set_seed(123)
np.random.seed(123)
rn.seed(123)

# Load Data
path = 'C:\\ai5\\_data\\dacon\\생명연구\\'
np_path = 'C:\\ai5\\_data\\_save_npy\\dacon02\\'
w_path = 'C:\\ai5\\_save\\dacon02\\'

x = joblib.load(np_path + 'x_data_13.dat')
y = joblib.load(np_path + 'y_data_13.dat')
test = joblib.load(np_path + 'test_data_13.dat')
# print(x.shape, y.shape, test.shape) # (6201, 4384) (6201,) (2546, 4384)
# print(y.value_counts())             # 데이터 불균형 있음

########## 클래스가 1개짜리 컬럼 삭제 ##########
# 각 컬럼의 클래스 개수 계산
class_counts = {col: len(np.unique(x[col])) for col in x.columns}

# 클래스 개수로 정렬
sorted_class_counts = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)

# 정렬된 결과 출력
for col, count in sorted_class_counts:
    print(f"Column : {col}, Number of Classes: {count}") # 최소 클래스 1개, 최대 클래스 664개

# 클래스 개수가 1개인 컬럼의 갯수 출력
columns_with_1_classes = [col for col, count in class_counts.items() if count == 1]
print(f"Number of columns with exactly 1 unique classes: {len(columns_with_1_classes)}")    # 154

# 클래스 개수가 1개인 컬럼 삭제
columns_with_1_classes = [col for col, count in class_counts.items() if count == 1]
x = x.drop(columns=columns_with_1_classes)
test = test.drop(columns=columns_with_1_classes)

# print(x, x.shape)       # (6201, 4230) // 4384 -> 4230
# print(test, test.shape) # (2546, 4230)

le_subclass = LabelEncoder()
y = le_subclass.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y, random_state=123)

model = xgb.XGBClassifier(
    n_estimators=300,           # epochs, 300
    learning_rate=0.12,         # 0.01 ~ 0.2, 후반에 설정
    max_depth=2,                # 2
    random_state=123,           # 123
    use_label_encoder=False,
    eval_metric='mlogloss',
    gamma=2,                    # 2
    # colsample_bytree=0.5,
)

start = time.time()
model.fit(x_train, y_train)
end = time.time()

joblib.dump(model, './_save/dacon02/d02_13.pkl')

y_predict = model.predict(x_test)
f1 = f1_score(y_test, y_predict, average = 'macro')

result = model.predict(test)
result = le_subclass.inverse_transform(result)

submisson = pd.read_csv(path + "sample_submission.csv")
submisson["SUBCLASS"] = result
submisson.to_csv(path + 'baseline_submission_13.csv', encoding='UTF-8-sig', index=False)

print('f1_score :', f1, '/ time :', round(end-start, 2), '초 >')

