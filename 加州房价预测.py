from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
data=fetch_california_housing()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

import torch
#转换为pytorch张量
x_train=torch.tensor(x_train,dtype=torch.float32)
x_test=torch.tensor(x_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.float32)
y_test=torch.tensor(y_test,dtype=torch.float32)

#增加一个维度以匹配网络输入
x_train=x_train.unsqueeze(1)
x_test=x_test.unsqueeze(1)
print(x_train.size())

import torch.nn as nn
class house_model(nn.Module):
    def __init__(self,num_features):
        super(house_model,self).__init__()
        self.fc1=nn.Linear(num_features,64)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,1)
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        return x
num_features=x_train.size()[2]
model=house_model(num_features)
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0.01)

print(model(x_train).size())
print(y_train.size()) #torch.Size([14448])
for i in range(1000):
    output=model(x_train)
    loss=criterion(output.squeeze(),y_train)
    #开始优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1)%100==0:
        print(f'epoch[{i+1}/{1000}],loss={loss.item():.4f}')
model.eval()#评估模式
from sklearn.metrics import r2_score
with torch.no_grad():
    pred=model(x_test)
    test_loss=criterion(pred.squeeze(),y_test)
    score=r2_score(y_test.numpy(),pred.squeeze().numpy())
    print(test_loss.item())
    print(score)
