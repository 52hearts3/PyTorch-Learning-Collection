from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader,TensorDataset
from torch import nn,functional,optim
from sklearn.metrics import r2_score
import numpy as np

data_in=fetch_california_housing()
data=data_in.data
target=data_in.target
x_train,x_test,y_train,y_test=train_test_split(data,target)

x_train=torch.tensor(x_train,dtype=torch.float32).unsqueeze(1).unsqueeze(1) #将数据数量转换为图片数量
x_test=torch.tensor(x_test,dtype=torch.float32).unsqueeze(1).unsqueeze(1)
y_train=torch.tensor(y_train,dtype=torch.float32).unsqueeze(1) #[数据量]-->[图片数量,通道]
y_test=torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)
print(y_train.shape)

train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)

train_loader=DataLoader(train_dataset,batch_size=64)
test_loader=DataLoader(test_dataset,batch_size=64)

class ResBlk(nn.Module):  #定义短接类
    def __init__(self,ch_in,ch_out):
        super(ResBlk,self).__init__()
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2=nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(ch_out)

        #判断输入的x是否可以和卷积后的数据相加
        self.extra=nn.Sequential(
            nn.Conv2d(ch_in,ch_in,kernel_size=1,stride=1),#kernel大小相等情况
            nn.BatchNorm2d(ch_in)
        )
        if ch_in != ch_out:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self,x):
        out=nn.functional.relu(self.bn1(self.conv1(x)))
        out=nn.functional.relu(self.bn2(self.conv2(out)))
        out=self.extra(x)+out
        return out
#测试
x=torch.randn(300,1,1,8)
model=ResBlk(1,64)
print(model(x).shape)
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,64,stride=1,kernel_size=3,padding=1),
            nn.BatchNorm2d(64)
        )
        self.blk1=ResBlk(64,128)
        self.blk2=ResBlk(128,256)
        self.blk3=ResBlk(256,512)
        self.blk4=ResBlk(512,512)

        self.linear=nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self,x):
        x=self.conv1(x)
        x=nn.functional.relu(x)
        x=self.blk1(x)
        x=self.blk2(x)
        x=self.blk3(x)
        x=self.blk4(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x
#测试
model=ResNet()
x=torch.randn(300,1,1,8)
test=model(x)
print(test.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet().to(device)
criteon = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) #动态调整学习率

for epoch in range(1000):
    model.train()
    for x, label in train_loader:
        x, label = x.to(device), label.to(device)
        logits = model(x)
        loss = criteon(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch, loss.item())

    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        # 确保预测值和真实值的形状一致
        all_preds = all_preds.flatten()
        all_labels = all_labels.flatten()
        score = r2_score(all_labels, all_preds)
        print(f'R² Score: {score}')





