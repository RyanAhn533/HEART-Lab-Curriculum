import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE : ', DEVICE)

# 1. 데이터
x = np.array([1, 2, 3], dtype=np.float32)
y = np.array([1, 2, 3], dtype=np.float32)

# 평균과 표준편차 계산
mean_x = np.mean(x)
std_x = np.std(x)

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
x = (x - mean_x) / std_x  # 정규화
print(x)

y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
print(y)

print(x.shape, y.shape)
print(x.size(), y.size())

# 2. 모델구성
model = nn.Linear(1, 1)  # 인풋, 아웃풋 

# 3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()  # 기울기 초기화

    hypothesis = model(x)  # y = wx + b
    loss = criterion(hypothesis, y)  # loss 계산
    
    loss.backward()  # 역전파
    optimizer.step()  # 가중치 업데이트
    
    return loss.item()

epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    if epoch % 100 == 0:  # 100 epoch마다 출력
        print('epoch: {}, loss : {}'.format(epoch, loss))

# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()  # 평가모드
    
    with torch.no_grad():
        y_predict = model(x)
        loss = criterion(y_predict, y)  # 예측값과 실제값 비교
    return loss.item()

loss2 = evaluate(model, criterion, x, y)
print("최종 loss :", loss2)

# 예측값
results = model(torch.Tensor([[4]]).to(DEVICE))  # 4의 예측값을 구하기 위해 텐서를 DEVICE에 맞춤
print('4의 예측값 : ', results.item())  # .item()으로 스칼라 값 추출
