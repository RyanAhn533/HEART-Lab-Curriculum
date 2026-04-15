#train_test_split 후 스케일링 후 PCA
#고쳐봐!!

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)


# 통상적으로 선 scaler 후 pca가 정석
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.9, random_state=5656, shuffle=True, stratify=y)
print(y_train.shape)
print(y_test.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=3) #3차원으로 줄인다 그냥 벡터에다가 쳐밖는다
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


#2.모델
model = RandomForestClassifier(random_state=5656)

model.fit(x_train,y_train)

results = model.score(x_test, y_test)
print('model.score는 ', results)

evr = pca.explained_variance_ratio_
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()