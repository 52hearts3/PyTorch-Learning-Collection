import numpy as np
from sklearn.datasets import load_iris
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from torch import nn,optim
from sklearn.metrics import accuracy_score

data_in=load_iris()
data=data_in.data
target=data_in.target
print(np.unique(target))  #三分类
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2)

x_train=torch.tensor(x_train,dtype=torch.float32).unsqueeze(1).unsqueeze(1)
x_test=torch.tensor(x_test,dtype=torch.float32).unsqueeze(1).unsqueeze(1)
y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)
print(y_test.shape)

train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)

train_loader=DataLoader(train_dataset,batch_size=32)
test_loader=DataLoader(test_dataset,batch_size=32)

#定义短接类
class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(ResBlk,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.extra=nn.Conv2d(ch_in,ch_in,kernel_size=1,stride=1,padding=0) #ch_in=ch_out
        if ch_in != ch_out:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,stride=1,kernel_size=1,padding=0),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self,x):
        out=self.conv(x)
        out=self.extra(x)+out
        return out
#test
test=torch.randn(32,1,1,8)
model=ResBlk(1,32)
print(model(test).shape)

#定义ResNet类
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.blk1=ResBlk(64,128)
        self.blk2=ResBlk(128,256)
        self.blk3=ResBlk(256,512)
        self.blk4=ResBlk(512,512)

        self.linear=nn.Linear(512,3)

    def forward(self,x):
        x=self.conv1(x)
        x=self.blk1(x)
        x=self.blk2(x)
        x=self.blk3(x)
        x=self.blk4(x)
        x=nn.functional.adaptive_avg_pool2d(x,(1,1))
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x
#测试
test=torch.randn(32,1,1,8)
model=ResNet()
print(model(test).shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet().to(device)
criteon=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)

model.train()
for epoch in range(1000):
    for batch_idx,(x,label) in enumerate(train_loader):
        x,label=x.to(device),label.to(device)
        logits=model(x)  #[b,3]  三分类
        #print(logits.shape)  #label.shape--[b]
        #print(label.shape)
        #MSE的输入和目标的形状必须相同，而交叉熵可以处理logits.shape=[b,3],label.shape=[b]的情况
        loss=criteon(logits,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(epoch,loss.item())
    #test
    model.eval()
    with torch.no_grad():  #对于分类问题，神经网络的预测结果并不是标签，而是不同标签的概率分布
        total_correct = 0
        total_sum = 0
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred, label).float().sum().item()  # 使用item转换为numpy
            total_sum += x.size(0)
        acc = total_correct / total_sum
        print(acc)
