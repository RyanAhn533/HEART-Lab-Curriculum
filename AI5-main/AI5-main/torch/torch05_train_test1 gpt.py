import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

# 데이터 정의
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([8,9,10,11,12,13,14])  # y_train과 y_test 길이 맞춤
x_test = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10,11,12,13,14])
x_predict = np.array([12,13,14])

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
x_predict = torch.FloatTensor(x_predict).unsqueeze(1).to(DEVICE)

# 모델 정의
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
).to(DEVICE)

# 손실 함수 및 최적화 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 훈련 함수
def train(model, criterion, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

# 훈련 루프
epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 100 == 0 or epoch == 1:
        print(f'epoch: {epoch}, loss: {loss}')

# 평가 함수
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_predict = model(x_test)
        loss = criterion(y_predict, y_test)
    return loss.item()

# 최종 손실 및 예측
loss2 = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', loss2)

# 새로운 값 예측
results = model(torch.Tensor([[10]]).to(DEVICE))
print('[10]의 예측값 :', results.detach().cpu().numpy())
