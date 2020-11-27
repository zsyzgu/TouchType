import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import random
import numpy as np
print(torch.__version__)

BATCH_SIZE=64
EPOCHS=100
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__() # 51 * 21
        self.conv1=nn.Conv2d(4, 10, 5) # 2 * 47 * 17 --> 2 * 23 * 8
        self.conv2=nn.Conv2d(10, 20, 3) # 10 * 21 * 6
        self.fc1=nn.Linear(20 * 21 * 6, 100)
        self.fc2=nn.Linear(100, 2)

    def forward(self, x):
        in_size=x.size(0)		# in_size 为 batch_size（一个batch中的Sample数）
        # 卷积层 -> relu -> 最大池化
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        #卷积层 -> relu -> 多行变一行 -> 全连接层 -> relu -> 全连接层 -> sigmoid
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(in_size, -1)     # view()函数作用是将一个多行的Tensor,拼接成一行。
        out = self.fc1(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

class Loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        [self.X, self.Y, self.Z] = pickle.load(open('data.pickle', 'rb'))
        self.X = [np.reshape(x,(4,51,21)) for x in self.X]

        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        test_users = ['swn', 'plh', 'grc', 'hxz']
        for x, y, z in zip(self.X, self.Y, self.Z):
            if y != -1:
                if z in test_users:
                    self.X_test.append(x)
                    self.Y_test.append(y)
                else:
                    self.X_train.append(x)
                    self.Y_train.append(y)
        
        self.X_train, self.Y_train = self.balance(self.X_train, self.Y_train)
        self.X_test, self.Y_test = self.balance(self.X_test, self.Y_test)

    def balance(self, X, Y):
        a = sum(np.array(Y) == 0)
        b = sum(np.array(Y) == 1)
        while a != b:
            i = random.randint(0, len(X) - 1)
            x = X[i]
            y = Y[i]
            flag = False
            if y == 0 and a < b:
                flag = True
                a += 1
            if y == 1 and b < a:
                flag = True
                b += 1
            if flag:
                X.append(x)
                Y.append(y)
        return X, Y

    def getData(self, X, Y):
        randnum = random.randint(0,100)
        random.seed(randnum)
        random.shuffle(X)
        random.seed(randnum)
        random.shuffle(Y)
        data = []
        i = 0
        for st in range(0, len(X), self.batch_size):
            en = min(st+self.batch_size, len(X))
            data.append((i, (torch.tensor(X[st:en]).type(torch.FloatTensor), torch.tensor(Y[st:en]).type(torch.LongTensor))))
            i += 1
        return data
    
    def getTrainData(self):
        return self.getData(self.X_train, self.Y_train)

    def getTestData(self):
        return self.getData(self.X_test, self.Y_test)

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())
loader = Loader(BATCH_SIZE)

def train(model, device, loader, optimizer, epoch):
    model.train()
    dataset = loader.getTrainData()
    for batch_idx, (data, target) in dataset:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    print('Train Epoch: %d    Loss: %f' % (epoch, loss.item()))

def test(model, device, loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        dataset = loader.getTestData()
        for batch_idx, (data, target) in dataset:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, len(loader.X_test),100. * correct /len(loader.X_test)))

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, loader, optimizer, epoch)
    test(model, DEVICE, loader)
