import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, dataloader
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE', DEVICE)

import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor()]) # minmax 정규화 값에서 0.5뺀 값서 0.5로 나눠줌

#1. 데이터
path = './study/torch/_data'
#train_dataset = MNIST(path, train=True, download=False)
#test_dataset = MNIST(path, train=True, download=False)

train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)


print(train_dataset[0][0].shape) #<PIL.Imahe
print(train_dataset[0][1])

##### 정규화 (Minmax) /255
# x_train, y_train = train_dataset.data/255. , train_dataset.targets
# x_test, y_test = test_dataset.data/255. , test_dataset.targets

### x_train/127.5 -1 #얘의 값의 범우는 #-1 ~ 1 #정규화 보다는 표준화에 가까움


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

###################### 잘 받아졌는지 확인 #######
bbb = iter(train_dataset)

aaa = next(bbb)
print(aaa)
print(aaa[0].shape)
print(len(train_loader))

x_train, y_train = train_dataset.data/255, train_dataset.targets
x_test, y_test = test_dataset.data/255, test_dataset.targets


#2. 모델
class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),
            # model.COnv2D(64, (3,3), stirde=1, input_shape=(56,56,1))
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), # (n, 64, 27, 27)
            nn.Dropout(0.5,)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64,32, kernel_size=(3,3), stride=1), #(n, 32, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),  #(n, 32, 12, 12)
            nn.Dropout(0.5,)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32,16, kernel_size=(3,3), stride=1), #(n, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5,)
        )
        
        self.hidden_layer4 = nn.Linear(16*5*5, 16)       
        self.output_layer = nn.Linear(in_features=32, out_features=10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = x.view(x.shape[0], -1)
        # x = flatten()(x) # 케라스에선 위에꺼를 욜케 썻지.
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x
    
model = CNN(1).to(DEVICE)
#3. 컴파일, 훈련

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4) #0.0001

def train(model, criterion, optimizer, loader):
    
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        hypothesis = model(x_batch)
        
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        epoch_acc += acc.item()
        
    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, criterion, loader):
    model.eval()
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad() :
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
            # 평가에는 순전파만 사용하면 된다 왜냐? 가중치 갱신을 할 필요가 없다
            
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            epoch_acc += acc.item()
        return epoch_loss / len(loader), epoch_acc / len(loader)

epochs = 20
for epoch in range(1, epochs + 1):
    #epochs +1로 한 이유는 가독성때문에 -> 0부터 시작하면 가독성 떨어짐
    loss, acc = train(model, criterion, optimizer, train_loader)
    
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    print('epoch: {}, loss:{:.4f}, acc:{:.3f}, val_loss{:.4f}'.format(epoch, loss, acc, val_loss, val_acc))
    

#4. 평가 예측
def evaluate(model, criterion, loader):
    model.eval()
    
    total_loss = 0
    for x_batch, y_batch in loader:
        with torch.no_grad():
            y_predict = model(x_batch)
            loss2 = criterion(y_predict, y_batch)
            total_loss += loss2.item()
    return total_loss / len(loader)

last_loss = evaluate(model, criterion, x_test, y_test)