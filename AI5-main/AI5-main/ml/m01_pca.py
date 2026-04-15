from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
#자동으로 accuracy를 뽑아준다 - RandomForestClassifier, Regressor = R2스코어
#RandomForest에서 분류를 할 꺼면 classfier 회귀할꺼면 Regressor
from sklearn.decomposition import PCA
#PCA decomposition(압축)
# 분포되어 있는 2차원 데이터를 벡테로 바꿔주는 것 
# -> 유실이 많다(성능 95%까지 떨어질 수 있음)
# pca = 차원축소


#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=1) #3차원으로 줄인다 그냥 벡터에다가 쳐밖는다
x = pca.fit_transform(x)
print(x)
print(x.shape) #(150, 3)

# 통상적으로 선 scaler 후 pca가 정석

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.9, random_state=5656, shuffle=True, stratify=y)
#분류에서는 꼭 해줘야함 stratify=y 
# y의 라벨의 비율을 맞추어서 train test 비율을 정해준다

#2.모델
model = RandomForestClassifier(random_state=5656)
#decision  tree 확장형 버전 = decision tree 앙상블 버전

#3. 훈련(머신러닝에는 컴파일 훈련을 안해도된다)
model.fit(x_train,y_train)

results = model.score(x_test, y_test)
print('model.score는 ', results)

#(150, 4) randomstate 7 
#model.score는  1.0
#(150, 3) randomstate 7
#model.score는  1.0
#5656 2 1.0
#5656 1 1.0