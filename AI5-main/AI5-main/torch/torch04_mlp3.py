import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda'if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

x = np.array([range(10), range(21,31), range(201, 211)]).transpose()
#1. 데이터
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [10,9,8,7,6,5,4,3,2,1]]).transpose()  #벡터 2개짜리는 행렬 ->
             #이렇게되면 행열이 반대로 찍힘 

#맹그러봐
print(x.shape, y.shape)

# 예측값 : [10, 1.3, 1]

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)


model = nn.Sequential(
    nn.Linear(3, 3),
    nn.Linear(3,2),
    nn.Linear(2,3)
).to(DEVICE)

#3. 컴파일
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#4. 훈련함수
def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

#5. 훈련 루프
epochs = 2000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))

def evaluate(model, criterion, x, y):        
    model.eval()
        
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
        
#7. 최종 손실과 예측 출력
loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)

#8. 새로운 값 예측
results = torch.Tensor([[10, 31, 211]]).to(DEVICE)
print('[10,31,211]의 예측값 :', results.detach().cpu().numpy())

#print(results)
#tensor([[ 10.,  31., 211.]], device='cuda:0')

#print(results.detach())
#tensor([[ 10.,  31., 211.]], device='cuda:0')

#print(results.detach().numpy())
#TypeError: can't convert cuda:0 device type tensor to numpy. 
# Use Tensor.cpu() to copy the tensor to host memory first.
#numpy는 cpu에서만 돌아서 에러다.

print(results.detach().cpu().numpy())
