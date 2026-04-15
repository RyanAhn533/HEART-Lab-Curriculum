import torch as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu' )
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=369, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE) #실수형으로 변환
# x_train = torch.DoubleTensor(x_train).to(DEVICE)

x_test = torch.FloatTensor(x_test).to(DEVICE)
# x_test = torch.DoubleTensor(x_test).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.LongTensor(y_train).unsqueeze(1),to(DEVICE) 
# y_train = torch.IntTensor(y_train).unsqueeze(1),to(DEVICE) #정수형으로 변환

y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
# y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)
# y_test = torch.IntTensor(y_test).unsqueeze(1).to(DEVICE)

print('======================================================')
print(x_train.shape, x_test.shape) #torch.Size([398, 30]) torch.Size([171, 30])
print(y_train.shape, y_test.shape) #torch.Size([398, 1]) torch.Size([171, 1])
print(type(x_train), type(y_train)) #<class 'torch.Tensor'> <class 'torch.Tensor'>

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.Linear(16, 1),
#     nn.Sigmoid()
# ).to(DEVICE)

class Model(nn.Module): #클래스 선언
    def __init__(self, input_dim, output_dim):
        # super().__init__() #nn.Midule에 있는 함수를 상속받아서 모두 쓰겠다. super는 default이다. 
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    #순전파!      
    def forward(self, input_size): #정의된 변수를 순서대로 정렬해 줌
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x         

model = Model(30, 1).to(DEVICE) #클래스 인스턴스화 #선언해준 input_dim과 outpu_dim 삽입

#3. 컴파일 훈련

criterion = nn.BCELoss() 

optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    # model.train() # 훈련모드, 디폴트
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward() ##기울기 gradient 값 계산까지, 역전파 시작! 이거 안하면 처음의 weight로 계속 
    optimizer.step() #가중치(w) 갱신 #역전파 끝
    return loss.item()

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss : {}'.format(epoch, loss)) #verbose

print('==================================================================')

#4. 평가, 예측
# loss = model.evaluate(x,y)

def evaluate(model, criterion, x, y):
    model.eval() #평가모드 // 역전파 x, 가중치 갱신 x, 기울기 계산 할 수도 안 할 수도 
                 # 드롭아웃, 배치노말 x
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss : ', last_loss)

y_pred = model(x_test)

print(y_pred.shape)



############################################ 요 밑에 완성할 것 #################################################

from sklearn.metrics import accuracy_score, r2_score

y_test = torch.round(y_test)
y_pred = torch.round(y_pred)

y_test = y_test.detach().cpu().numpy()
y_pred = y_pred.detach().cpu().numpy()


# y_test = np.reshape(y_test, (171, 1))
# y_pred = np.reshape(y_pred, (171, 1))
print(y_test.shape)
print(y_pred.shape)

# print(type(y_test))

# y_test = np.round(y_test)
# y_pred = np.round(y_pred)



acc = accuracy_score(y_test, y_pred)

# acc = acc.detach().cpu().numpy()

print('acc 스코어 : ', acc)

