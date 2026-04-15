#데이터를 분리하고 싶다면 / 그냥 모델을 똑같이 두고 x, y데이터 열에
#구멍을 뻥뻥 뚫어놓고 예측 모델 돌리돌리기
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

path = "C:\프로그램\ai5\_data\bike-sharing-demand"
train_csv = pd.read_csv(path + "train_csv", index_col=0)
test_csv = pd.read_csv(path + "test_csv", index_col=0)

x = train_csv.drop(['casual', 'registerd', 'count'], axis =1)
y = train_csv[['casual', 'registered']]

print(x.info())
print(y.columns)