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
print(x_train.unsqueeze(1).unsqueeze(3).size())

from torch.utils.data import TensorDataset,DataLoader
#创建TrnsorDataset
train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)

#使用DataLoader批处理数据
train_loader=DataLoader(dataset=train_dataset,batch_size=128,shuffle=True,drop_last=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=128,shuffle=False,drop_last=True)


import torch.nn as nn
class house_model(nn.Module):
    def __init__(self,input_dim):
        super(house_model,self).__init__()
        self.fc1=nn.Conv2d(1,16,kernel_size=(3,1),padding=(1,0))
        self.relu=nn.ReLU()
        # 计算卷积层输出的维度
        self.con2=nn.Conv2d(16,32,kernel_size=(3,1),padding=(1,0))
        conv_output_dim = self.con2(torch.zeros(1, 16, input_dim, 1)).view(-1).shape[0]
        # 通过将全零张量传递给卷积层并展平输出,计算卷积层输出的维度
        self.li = nn.Linear(conv_output_dim, 32)
        self.fc2=nn.Linear(32,1)

    def forward(self,x):
        x = x.unsqueeze(1).unsqueeze(3)  # 增加通道维度和高度维度
        # 在输入张量x上增加通道维度和高度维度,以满足卷积层的输入要求
        x = self.fc1(x)
        x=self.relu(x)
        x=self.con2(x)
        x=self.relu(x)
        x = x.view(x.size(0), -1)
        # 将卷积层的输出展平为二维张量,第一维为批次大小,第二维为特征维度)
        x=self.li(x)
        x=self.fc2(x)
        return x
model = house_model(input_dim=x_train.shape[1])
# 创建CNN模型的实例,输入维度为X的特征数
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.01)

from sklearn.metrics import r2_score
for epoch in range(40):
    total_loss=0
    model.train()
    for inputs,target in train_loader:
       optimizer.zero_grad()
       output=model(inputs)
       loss=criterion(output,target.unsqueeze(1))#-r2_score(target.detach().numpy(),output.detach().numpy())
       #开始优化
       loss.backward()
       optimizer.step()
       total_loss+=loss.item()
    ave_loss=total_loss/len(train_loader)
    #if (epoch+1)%10==0:
    print(f'epoch[{epoch+1}/{40}],loss={ave_loss:.4f}')
model.eval()#评估模式
with torch.no_grad():
    score_all=[]
    score_mse=[]
    for inputs, target in test_loader:
      pred=model(inputs)
      test_loss=criterion(pred,target.unsqueeze(1))
      pred=pred.squeeze().numpy()
      score=r2_score(target.numpy(),pred)
      score_all.append(score)
      score_mse.append(test_loss.item())
      print(test_loss.item())
print('score :',sum(score_all)/len(score_all))
print('mse :',sum(score_mse)/len(score_mse))