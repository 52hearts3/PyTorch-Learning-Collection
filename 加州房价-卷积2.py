from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch import nn,optim
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import r2_score
import numpy as np
data_in=fetch_california_housing()
data=data_in.data
target=data_in.target
x_train,x_test,y_train,y_test=train_test_split(data,target)

x_train=torch.tensor(x_train,dtype=torch.float32)
x_test=torch.tensor(x_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.float32)
y_test=torch.tensor(y_test,dtype=torch.float32)

x_train=x_train.unsqueeze(1).unsqueeze(1)#将数据的条数看作通道，将数据的特征看作图片的宽度--[批次大小,通道,高,宽]
x_test=x_test.unsqueeze(1).unsqueeze(1)
y_train=y_train.unsqueeze(1)  #线性层输出结果为[b,1],故与输出结果保持一致
y_test=y_test.unsqueeze(1)
print(x_train.shape)

train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)
train_loader=DataLoader(dataset=train_dataset,batch_size=34)
test_loader=DataLoader(dataset=test_dataset,batch_size=34)

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.con=nn.Sequential(
            nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1),
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        )
        self.linear=nn.Sequential(
            nn.Linear(16*8,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,1)
        )
    def forward(self,x):
        x=self.con(x)
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x
def test():
    tmp = torch.randn(15480, 1, 1, 8)
    net_1=net()
    out=net_1.forward(tmp)
    print(out)
test()
model=net()
creterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)
for epoch in range(1000):
    model.train()
    for x, label in train_loader:
        logits=model(x)
        loss=creterion(logits,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch,loss.item())
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for x, label in test_loader:
            pred = model(x)
            all_preds.append(pred.numpy())
            all_labels.append(label.numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        score = r2_score(all_labels, all_preds)
        print(f'R² Score: {score}')




