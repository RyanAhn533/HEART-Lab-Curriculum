import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

#토치의 텐서형태
#x = torch.FloatTensor(x)
#y = torch.LongTensor(y)
#print(x.shape, y.shape)
#토치에서는 shape 대신에 size가 있다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x, y, shuffle=True, train_size=0.7, 
                                                     random_state=1004, stratify=y)

y_train = y_train - 1
y_test = y_test - 1

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)


#print(x_train.size(), y_train.size())
#print(x_test.size(), y_test.size())

#GPU 사용 설정
x_train = x_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_train = y_train.to(DEVICE)
y_test = y_test.to(DEVICE)

#2. 모델
model = nn.Sequential(
    nn.Linear(54, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 7),
).to(DEVICE)

#컴파일, 훈련
#이거쓰면 softmax안써도댐
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

#웨이트 = 웨이트 - 런닝레이트*로스/기울기

def train(model, criterion, optimizer, x_train, y_train):
    #model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward() #가중치 갱신이라는 말도 맞지만 정확한 표현은 기울기 계산
    optimizer.step()
    return loss.item()

EPOCHS = 1000 #대문자로 쓰는게 암묵적인 약속, 특정 상수를 이야기할 때 고정된 숫자 !
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    #print('epoch : {}, loss: {:.8f}'.format(epoch, loss))
    print(f'epoch:{epoch}, loss: {loss}')
    

#4. 평가 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()
    
loss = evaluate(model, criterion, x_test, y_test)
print('loss : ', loss)

########## acc 출력해봐용 #############

# from sklearn.metrics import accuracy_score
# result = model(x_test)
# result = np.round(model(x_test))

# acc = accuracy_score(y_test, result)
# print('acc : ?', acc)

print(model(x_test[:5]))

y_predict = model(x_test)
print(y_predict[:5])

y_predict = torch.argmax(y_predict, dim=1)

score = (y_predict == y_test).float().mean()
print('accuaracy : {:4f}'.format(score))
print(f'accuaracy : {score:.4f}')

score2 = accuracy_score(y_test.cpu().numpy(),
                        y_predict.cpu().numpy()) #넘파이는 cpu에서만 돌아간다.

"""
tensor([[ 10.8583,  11.9181, -24.0185, -43.5844, -20.8172, -14.7147,  -2.7201],
        [ 10.8664,   5.5702, -23.8222, -31.6819, -21.4541, -11.8626,   3.5132],
        [ 11.3695,  10.9987, -11.4765, -34.2579,  -4.1694,  -4.5208, -11.5734],
        [ -0.5374,   4.7969,   3.1103, -10.7183,   1.4813,  -1.4489,  -5.4739],
        [ 11.4248,   8.4379, -20.7652, -29.6472, -13.2729,  -8.0890,  -1.7048]],
       device='cuda:0', grad_fn=<AddmmBackward0>)
tensor([[ 10.8583,  11.9181, -24.0185, -43.5844, -20.8172, -14.7147,  -2.7201],
        [ 10.8664,   5.5702, -23.8222, -31.6819, -21.4541, -11.8626,   3.5132],
        [ 11.3695,  10.9987, -11.4765, -34.2579,  -4.1694,  -4.5208, -11.5734],
        [ -0.5374,   4.7969,   3.1103, -10.7183,   1.4813,  -1.4489,  -5.4739],
        [ 11.4248,   8.4379, -20.7652, -29.6472, -13.2729,  -8.0890,  -1.7048]],
       device='cuda:0', grad_fn=<SliceBackward0>)
accuaracy : 0.823825
accuaracy : 0.8238

"""