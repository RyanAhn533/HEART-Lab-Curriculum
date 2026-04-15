import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

SEED = 1004
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# 데이터 로드 및 전처리
datasets = fetch_covtype()
x = datasets.data
y = datasets.target - 1  # 클래스가 1부터 시작하므로 0부터 시작하도록 조정

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=369)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape, x_test.shape) #torch.Size([398, 30]) torch.Size([171, 1, 30])
print(y_train.shape, y_test.shape) #torch.Size([398]) torch.Size([171, 1])
print(type(x_train), type(y_train))

# 모델 정의
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.linear5(x)  # CrossEntropyLoss가 내부적으로 Softmax를 사용하므로 여기서 Softmax는 생략
        return x

model = Model(54, 7).to(DEVICE)

# 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {:.4f}'.format(epoch, loss))
    
# 평가
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_predict = model(x)
        loss = criterion(y_predict, y)
    return loss.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", last_loss)

# 정확도 계산
result = model(x_test)
result = torch.argmax(result, dim=1)  # 가장 높은 확률의 클래스 선택
acc = accuracy_score(y_test.cpu().numpy(), result.cpu().numpy())
print('acc는?', acc)
