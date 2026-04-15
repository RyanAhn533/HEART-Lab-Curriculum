import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda'if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10,11])
y_test = np.array([8,9,10,11])
x_predict = np.array([12,13,14])

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
x_predict = torch.FloatTensor(x_predict).unsqueeze(1).to(DEVICE)


model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Linear(32,16),
    nn.Linear(16,1)
).to(DEVICE)

#. 컴파일

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#4.훈련 함수
def train(model, criterion, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()


#5.훈련루프
epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 100 == 0 or epoch ==1:
        print('epoch: {}, loss: {}'.format(epoch, loss))


#6. 평가함수
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_predict = model(x)
        loss = criterion(y_predict, y)
    return loss.item()

#7. 최종 손실과 예측 출력
loss2 = evaluate(model, criterion,x_test , y_test)
print('최종 loss :', loss2)

#8. 새로운 값 예측
results = model(x_predict.to(DEVICE))
print('[10]의 예측값 :', results.detach().cpu().numpy())