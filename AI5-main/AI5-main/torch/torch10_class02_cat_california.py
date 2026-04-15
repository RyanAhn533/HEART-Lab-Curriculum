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

x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.LongTensor(x_test).to(DEVICE)
y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)



#y_predict -> vector형태라서 reshape해줘야함

#int - long
#floattensor - doubletensor
#longtensor랑 floattensor언제 쓰는지 확인

print("========================")
print(x_train.shape, x_test.shape) #torch.Size([398, 30]) torch.Size([171, 1, 30])
print(y_train.shape, y_test.shape) #torch.Size([398]) torch.Size([171, 1])
print(type(x_train), type(y_train)) #<class 'torch.Tensor'> <class 'torch.Tensor'> 

# #2. 모델구성
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 1),
#     nn.Sigmoid()    
# ).to(DEVICE)

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
from sklearn.metrics import r2_score
result = model(x_test)
r2 = r2_score(y_test.cpu().numpy(), np.round(result.detach().cpu().numpy()))
print('r2는?', r2)

'''
최종 loss :  0.3196646571159363
r2는? 0.7096512317657471
'''