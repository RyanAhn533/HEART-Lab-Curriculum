import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

##########################랜덤 고정하자꾸나 ####################
SEED = 1004

import random
random.seed(SEED)   #파이썬 랜덤 고정
np.random.seed(SEED)   #넘파이 랜덤 고정

##토치 시드 고정
torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
#(569, 30) (569,)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True,
                                                    random_state=369)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#############토치로 변환###################
x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

from torch.utils.data import TensorDataset #x, y 합친다
from torch.utils.data import dataloader #batch 정의 .

#토치 데이터셋 만들기 1. x와 y를 합친다.
train_set = TensorDataset(x_train, y_train) #합쳐지니까 tensor tuple형태
test_set = TensorDataset(x_test, y_test)
print(train_set)
print(len(train_set)) #398 = x_train과 y_train 행의 개수
print(train_set[0])
print(train_set[0][0]) # 첫번째 x
print(train_set[0][1]) # 첫번째 y train_set[397]까지 있겠지

#토치 데이터셋 만들기 2. batch 넣어준다. 끝

train_loader = dataloader(train_set, batch_size=40, shuffle=True)
test_loader = dataloader(train_set, batch_size=40, shuffle=False)
print(len(train_loader)) #batch 40개로 자르다보니 10개
print(train_loader)#<torch.utils.data.dataloader.DataLoader object at 0x00000236EC4C0A70>



# 지금까지 x y 합치고 batch 정의해주고 iterator의 정의를 만들었다.

bbb = iter(train_loader)
#aaa = bbb.next()  # 파이썬 3.9까지 먹히는 문법
aaa = next(bbb)
print(aaa)


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        #super().__init__() #명시해줘도 되고 안해줘도 되낟 이건 디폴트다
        super(Model, self).__init__()  #당분간은 그냥 하나의 덩어리로 써라
        self.linear1 = nn.Linear(input_dim,64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,16)
        self.linear5 = nn.Linear(16,output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        #순전파
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)    
    
        x = self.linear2(x)
        x = self.relu(x)    
        x = self.linear3(x)
        x = self.relu(x)    
        x = self.linear4(x)
        x = self.linear5(x)
        return(x)

model = Model(8, 1).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()  # 회귀 문제이므로 MSELoss 사용

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    #model.train() 훈련모드 디폴트
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)  #batch 단위로
        loss = criterion(hypothesis, y_batch)
    
        loss.backward()  #역전파 시작
        optimizer.step()
        total_loss += loss.item
    return total_loss / len(loader)

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
#4. 평가 예측
#loss = model.evaluate(x, y)
def evaluate(model, criterion, loader):
    model.eval()  #평가모드 // 역전파, 가중치 갱신, 기울기 계산 안하고싶지만 할수도있기도 없기도
                  #drop out / batch normalization
    total_loss = 0
    for x_batch, y_batch in loader:
        with torch.no_grad():
            y_predict = model(x_batch)
            loss2 = criterion(y_predict, y_batch)
            total_loss += loss2.item()
    return total_loss / len(loader)

last_loss = evaluate(model, criterion, x_test, y_test)
# print("최종 loss : ", last_loss)

# #########요 밑에 완성할 것 (데이터 로더를 사용하는 것으로 바꿔라) ####################
from sklearn.metrics import r2_score
def acc_score(model, loader):
    x_test = []
    y_test = []
    for x_batch, y_batch in loader:
        x_test.extend(x_batch.detach().cpu().numpy())
        y_test.extend(y_batch.detach().cpu().numpy())
    x_test = torch.FloatTensor(x_test).to(DEVICE)
    y_pre = model(x_test)
    acc = r2_score(y_test, np.round(y_pre.detach().cpu().numpy()))
    return acc

r2 = r2_score(model, test_loader)
print('r2_score :', r2)

# y_pred = model(x_test)
# acc = accuracy_score(y_pred, x_test)
# print("acc는 ? : {:.4f}", acc)

#1. 텐서데이터셋으로 합친다
#2. 데이터로더로 배치를 정의해준다
#train은 통상 shuffle true test는 shuffle false
#배치를 정의한다