import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
# pip install category_encoders
from category_encoders import TargetEncoder

path = 'C:\\ai5\\_data\\dacon\\생명연구\\'
train_csv = pd.read_csv(path + "train.csv")
test_csv = pd.read_csv(path + "test.csv")

# print(train_csv, train_csv.shape)         # (6201, 4386)
# print(test_csv, test_csv.shape)           # (2546, 4385) 

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

te = TargetEncoder(cols=categorical_columns)
x_encoded = x.copy()
test_csv_encoded = test_csv.copy()

start = time.time()
x_encoded = te.fit_transform(x_encoded, y)
test_csv_encoded = te.transform(test_csv_encoded)

-print(x_encoded)
'''
            A2M       AAAS      AADAT      AARS1       ABAT     ABCA1      ABCA2      ABCA3      ABCA4      ABCA5  ...     ZNF292     ZNF365     ZNF639     ZNF707      ZNFX1      ZNRF4       ZPBP       ZW10      ZWINT        ZYX
0     11.755824  11.797954  11.801652  11.817449  11.800748  11.78048  11.741569  11.758609  11.749625  11.771518  ...  11.755353  11.783722  11.806614  11.802883  11.767848  11.784493  11.785935  11.803973  11.813007  11.798831
1     11.755824  11.797954  11.801652  11.817449  11.800748  11.78048  11.741569  11.758609  11.749625  11.771518  ...  11.755353  11.783722  11.806614  11.802883  11.767848  11.784493  11.785935  11.803973  11.813007  11.798831
2     12.882068  11.797954  11.801652  11.817449  11.800748  11.78048  11.741569  11.758609  11.749625  11.771518  ...  11.755353  11.783722  11.806614  11.802883  11.767848  11.784493  11.785935  11.803973  11.813007  11.798831
3     11.755824  11.797954  11.801652  11.817449  11.800748  11.78048  11.741569  11.758609  11.749625  11.771518  ...  11.755353  11.783722  11.806614  11.802883  11.767848  11.784493  11.785935  11.803973  11.813007  11.798831
4     11.755824  11.797954  11.801652  11.817449  11.800748  11.78048  11.741569  11.758609  11.749625  11.771518  ...  11.755353  11.783722  11.806614  11.802883  11.767848  11.784493  11.785935  11.803973  11.813007  11.798831
...         ...        ...        ...        ...        ...       ...        ...        ...        ...        ...  ...        ...        ...        ...        ...        ...        ...        ...        ...        ...        ...
6196  11.755824  11.797954  11.801652  11.817449  11.800748  11.78048  11.741569  11.758609  11.749625  11.771518  ...  11.755353  11.783722  11.806614  11.802883  11.767848  11.784493  11.785935  11.803973  11.813007  11.798831
6197  11.755824  11.797954  11.801652  11.817449  11.800748  11.78048  11.741569  11.758609  11.749625  11.771518  ...  11.755353  11.783722  11.806614  11.802883  11.767848  11.784493  11.785935  11.803973  11.813007  11.798831
6198  11.755824  11.797954  11.801652  11.817449  11.800748  11.78048  11.741569  11.758609  11.749625  11.771518  ...  11.755353  11.783722  11.806614  11.802883  11.767848  11.784493  11.785935  11.803973  10.800332  11.798831
6199  11.755824  11.797954  11.801652  11.817449  11.800748  11.78048  11.741569  11.758609  11.749625  11.771518  ...  11.755353  11.783722  11.806614  11.802883  11.767848  11.784493  11.785935  11.803973  11.813007  11.798831
6200  11.755824  11.797954  11.801652  11.817449  11.800748  11.78048  11.741569  11.758609  11.749625  11.771518  ...  11.755353  11.783722  11.806614  11.802883  11.767848  11.784493  11.785935  11.803973  11.813007  11.798831
[6201 rows x 4384 columns]
'''

import joblib
joblib.dump(x_encoded, path + 'x_data_13.dat')
# joblib.dump(y, path + 'y_data.dat')                           # 이것은 라벨인코더 된 상태라서
joblib.dump(train_csv['SUBCLASS'], path + 'y_data_13.dat')      # 라벨인코딩 하기전으로 세이브
joblib.dump(test_csv_encoded, path + 'test_data_13.dat')

end = time.time()
print('time :', round(end - start, 2))

