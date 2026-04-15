import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import TensorDataset, DataLoader

random.seed(333)
np.random.seed(333)
torch.manual_seed(333) # CPU 고정
torch.cuda.manual_seed(333) # GPU 고정

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 한 줄로 변경
print(DEVICE)

# 1. 데이터 준비
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)
x = x.reshape(x.shape[0], x.shape[1], 1)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)
print(x.shape, y.size())

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# 2. 모델 정의
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.RNN(input_size=1, hidden_size=32, 
                           batch_first=True) 
        self.fc1 = nn.Linear(3*32, 16)  # (N, 3*32) -> (N, 16)
        self.fc2 = nn.Linear(16, 8)     # (N, 16) -> (N, 8)
        self.fc3 = nn.Linear(8, 1)      # (N, 8) -> (N, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.cell(x)  # RNN 레이어, 히든 상태는 사용하지 않음
        x = self.relu(x)
        x = x.reshape(-1, 3*32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = RNN().to(DEVICE)

# 모델 요약
from torchsummary import summary
summary(model, (3, 1))  # (time steps, input size) 형태로 입력 크기 지정
