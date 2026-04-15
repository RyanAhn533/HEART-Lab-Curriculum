import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

path = "./_data/kaggle/Bank/"

# CSV 파일 로드
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

# 데이터 확인
print(train_csv['Geography'].value_counts())

# 데이터 변환
train_csv['Geography'] = train_csv['Geography'].replace({'France': 1, 'Spain': 2, 'Germany': 3})
test_csv['Geography'] = test_csv['Geography'].replace({'France': 1, 'Spain': 2, 'Germany': 3})
train_csv['Gender'] = train_csv['Gender'].replace({'Male': 1, 'Female': 2})
test_csv['Gender'] = test_csv['Gender'].replace({'Male': 1, 'Female': 2})


# 특정 열에 0 값을 가진 행 삭제
"""
train_csv = train_csv[train_csv['Balance'] != 0]
test_csv = test_csv[test_csv['Balance'] != 0]
"""

# 문자열 값을 가진 열 확인 및 삭제
print(train_csv.select_dtypes(include=['object']).columns)
print(test_csv.select_dtypes(include=['object']).columns)

# 'Surname' 열 삭제
train_csv = train_csv.drop(['Surname'], axis=1)
test_csv = test_csv.drop(['Surname'], axis=1)

# 데이터 저장
train_csv.to_csv(path + "replaced_train.csv")
test_csv.to_csv(path + "replaced_test.csv")

# 데이터 로드
re_train_csv = pd.read_csv(path + "replaced_train.csv", index_col=0)
re_test_csv = pd.read_csv(path + "replaced_test.csv", index_col=0)

# 데이터 확인
re_train_csv.info()
re_test_csv.info()

# 특정 열 제거
re_train_csv = re_train_csv.drop(['CustomerId'], axis=1)
re_test_csv = re_test_csv.drop(['CustomerId'], axis=1)


# 데이터 스케일링
scaler = StandardScaler()
re_train_csv_scaled = scaler.fit_transform(re_train_csv.drop(['Exited'], axis=1))
re_test_csv_scaled = scaler.transform(re_test_csv)


# 데이터프레임으로 변환
re_train_csv = pd.concat([pd.DataFrame(re_train_csv_scaled), re_train_csv['Exited'].reset_index(drop=True)], axis=1)
re_test_csv = pd.DataFrame(re_test_csv_scaled)



# 학습 데이터 분리
x = re_train_csv.drop(['Exited'], axis=1)
y = re_train_csv['Exited']
print(x.shape, y.shape)
#(569, 30) (569,)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True,
                                                    random_state=369, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
#x_train = torch.DoubleTensor(x_train).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
#y_train = torch.IntTensor(y_train).unsqueeze(1).to(DEVICE)
#y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)

x_test = torch.FloatTensor(x_test).to(DEVICE)
#x_test = torch.DoubleTensor(x_test).unsqueeze(1).to(DEVICE)

y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
#y_test = torch.IntTensor(y_test).unsqueeze(1).to(DEVICE)
#y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)


#y_predict -> vector형태라서 reshape해줘야함

#int - long
#floattensor - doubletensor
#longtensor랑 floattensor언제 쓰는지 확인

print("========================")
print(x_train.shape, x_test.shape) #torch.Size([398, 30]) torch.Size([171, 1, 30])
print(y_train.shape, y_test.shape) #torch.Size([398]) torch.Size([171, 1])
print(type(x_train), type(y_train)) #<class 'torch.Tensor'> <class 'torch.Tensor'> 

#2. 모델구성
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()    
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()  #역전파 시작
    optimizer.step()
    return loss.item()

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
#4. 평가 예측
#loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()  #평가모드 // 역전파, 가중치 갱신, 기울기 계산 안하고싶지만 할수도있기도 없기도
                  #drop out / batch normalization
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y_predict, y)
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", last_loss)

##############################요 밑에 완성할 것 ####################
from sklearn.metrics import accuracy_score
result = model(x_test) #y_pred
result = np.round(model(x_test)) #y_pred

acc = accuracy_score(y_test.cpu().numpy(), np.round(result.detach().cpu().numpy()))
print('acc는?', acc)

#최종 loss :  0.5887978672981262
#acc는? 0.9941520467836257

y_test = y_test.cpu().numpy()

acc = accuracy_score(y_test, result)
print('acc : ?', acc)