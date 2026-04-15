import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchsummary import summary

# Seed 설정
random.seed(333)
np.random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed(333)

# 장치 설정
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# 데이터 생성
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])

y = np.array([4,5,6,7,8,9,10])

x = x.reshape(x.shape[0], x.shape[1], 1)
x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)

# 데이터 로더 설정
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# 모델 정의
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.LSTM(input_size=1, 
                            hidden_size=32, 
                            num_layers=1, 
                            batch_first=True)
        self.fc1 = nn.Linear(3*32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x, h0=None):
        if h0 is None:
            h0 = torch.zeros(1, x.size(0), 32).to(DEVICE)
        x, _ = self.cell(x, h0)
        x = self.relu(x)
        x = x.contiguous().view(-1, 3*32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 모델 인스턴스 생성 및 요약 출력
model = LSTM().to(DEVICE)
summary(model, (3, 1))


#컴파일, 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, criterion, optimizer, loader):
    epoch_loss = 0  # 여기에 초기화 추가
    
    model.train()
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)
        
        optimizer.zero_grad()
        h0 = torch.zeros(1, x.batch_size(0),32).to(DEVICE) # (num_layers, batch_size, hidden_size)
        hypothesis = model(x_batch, h0)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward() # 기울기 계산
        optimizer.step() # 가중치 갱신
        
        epoch_loss += loss.item()  # 손실 누적        
    return epoch_loss / len(loader)  # 평균 손실 반환



def evaluate(model, criterion, loader):
    epoch_loss = 0
    model.eval()
    
    with torch.no_grad() :
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1,1)
            h0 = torch.zeros(1, x.size(0),32).to(DEVICE) # (num_layers, batch_size, hidden_size)
            
            hypothesis = model(x_batch, h0)
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(loader)
    

for epoch in range(1, 1001):
    loss = train(model, criterion, optimizer, train_loader)
    
    if epoch %20 == 0 :  #20 에포마다 학습결과 출력
        print('epoch: {}, loss: {}'.format(epoch, loss))

x_predict = np.array([[8, 9, 10]])

def predict(model, data):
    model.eval()
    with torch.no_grad():
        data = torch.FloatTensor(data).unsqueeze(2).to(DEVICE) # (1, 3) ->  (1, 3, 1)
        h0 = torch.zeros(1, data.size(0),32).to(DEVICE) # (num_layers, batch_size, hidden_size)
        
        y_predict = model(data, h0)
    return y_predict.cpu().numpy()

y_predict = predict(model, x_predict)
print("===============================================")
print(y_predict)
print("===============================================")
print(y_predict[0])
print("===============================================")
print(f'{x_predict}의 예측값은 : {y_predict[0][0]}')
