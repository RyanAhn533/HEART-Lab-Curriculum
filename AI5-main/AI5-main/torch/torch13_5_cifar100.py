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

path = 'C:\\프로그램\\ai5\\study\\torch\\'
transform = transforms.ToTensor()

train_dataset = CIFAR100(path, train=True, download=True, transform=transform)
test_dataset = CIFAR100(path, train=False, download=True, transform=transform)

print(train_dataset)
print(type(train_dataset))
print(train_dataset[0][0])

bbb = iter(train_dataset)
aaa = next(bbb)
print(aaa)

x_train, y_train = train_dataset.data/255, train_dataset.targets
x_test, y_test = test_dataset.data/255, test_dataset.targets

print(x_test, x_train, x_train.shape, y_train.size())

print(np.min(x_train.numpy)), np.max(x_train.numpy)

x_train, x_test = x_train.view(-1, 3*32*32), x_test.reshape(-1, 3*32*32)
print(x_train.shape, x_test.size())

train_dataset = TensorDataset(x_train, y_train)
test_loader = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.output_layer = nn.Linear(32, 100)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        return x
    
model = DNN(784).to(DEVICE)
    
#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
#categorical 대신 sparse catergory가 들어간 상태
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, criterion, optimizer, loader):
    epoch_loss=0
    epoch_acc=0
    
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
            
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            epoch_acc += acc.item()
        return epoch_loss / len(loader), epoch_acc / len(loader)
    
epochs = 20

for epoch in range(1, epochs +1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print('epoch: {}, loss:{:.4f}, acc:{:.3f}, val_loss{:.4f}'.format(epoch, loss, acc, val_loss, val_acc))
    
#4. 평가예측
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