import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(datasets)

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)
df['Target'] = y
print(df)

print("====================== 상관계수 히트맵 ====================")
print(df.corr())

'''
====================== 상관계수 히트맵 ====================
                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
Target                      0.782561         -0.426658           0.949035          0.956547  1.000000
'''

#-0.99
#사랑의 반댓말은 미움이 아니라 무관심이다 = 0에 가까울수록 더 안좋다
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
print(sns.__version__)
print(matplotlib.__version__)
#sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square=True,
            annot=True,
            cbar=True)
plt.show()