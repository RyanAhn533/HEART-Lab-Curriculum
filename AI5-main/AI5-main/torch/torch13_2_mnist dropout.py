import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE', DEVICE)

path = 'C:\\프로그램\\ai5\\study\\torch\\'
transform = transforms.ToTensor()

train_dataset = MNIST(path, train=True, download=True, transform=transform)
test_dataset = MNIST(path, train=False, download=True, transform=transform)

print(train_dataset)
print(type(train_dataset))
print(train_dataset[0][0])

bbb = iter(train_dataset)
# aaa = bbb.next()
aaa = next(bbb)
print(aaa)

x_train, y_train = train_dataset.data/255, train_dataset.targets
x_test, y_test = test_dataset.data/255, test_dataset.targets

print(x_train)
print(y_train)
print(x_train.shape, y_train.size()) #torch.Size([60000, 28, 28]) torch.Size([60000])

print(np.min(x_train.numpy)), np.max(x_train.numpy) #0.0 1.0

x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784)
print(x_train.shape, x_test.size()) # torch.Size([60000, 784]) torch.Size([10000, 784])

train_dataset = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

#2. 모델
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
    
model = DNN(784).to(DEVICE)

#3. 컴파일 훈련
criterion = nn.CrossEntropyLoss()
#categorical 대신 sparse category 가 들어간 상태

optimizer = optim.Adam(model.parameters(), lr=1e-4) #0.0001

def train(model, criterion, optimizer, loader):
    #model.train() 가중치 갱신을 할 수도 있고 안 할 수도 있다
    epoch_loss = 0
    epoch_acc = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        hypothesis = model(x_batch) # y = wx + b
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        # acc = True False .float(0 or 1) . mean
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
    #loss, acc = model.evaluate(x_test, y_test)
    
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