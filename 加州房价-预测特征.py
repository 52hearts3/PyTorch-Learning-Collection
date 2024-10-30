from sklearn.datasets import fetch_california_housing
import torch
from torch import nn,optim
from torch.utils.data import DataLoader,TensorDataset
data_in=fetch_california_housing()
data=data_in.data
target=data_in.target
print(data)

#考虑到时间，不能随机分训练集，测试集
L_train=int(data.shape[0]*0.8)
x_train=data[:L_train]
y_train=target[:L_train]
print(x_train.shape)

x_train=torch.tensor(x_train,dtype=torch.float32).unsqueeze(1)
x_test=torch.tensor(data[1:L_train+1],dtype=torch.float32).unsqueeze(1)
y_train=torch.tensor(y_train,dtype=torch.float32).unsqueeze(1).unsqueeze(1)
y_test=torch.tensor(target[1:L_train+1],dtype=torch.float32)
print(x_train.shape)
print(y_train.shape)

#我们的输入数据为3维，[x,y,z]
# x是时间步，也就是每个序列的长度
# y是序列个数，也就是我们希望同时处理多少个序列
# z是输入数据的维度，也就是对于每个时间序列，每一天的数据维度
# 对于本问题，y_train的shape为(16512)，即x是16512，z是1

#使用标签预测特征
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(LSTM,self).__init__()
        self.LSTM=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
        # input_size可以理解为每天的数据维度，我们这里的特征为1
        self.linear=nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,8)
        )

    def forward(self,x):
        out,(h,c)=self.LSTM(x)
        seq_len,batch_size,hid_dim=out.shape
        out=out.view(-1,hid_dim)
        out=self.linear(out)
        out=out.view(seq_len,batch_size,-1)
        #print(out.shape)
        return out
x=torch.randn(32,1,1)
model=LSTM(1,10,1)
test=model(x)
print(test.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=LSTM(1,10,1).to(device)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=1e-4)
model.train()
for i in range(50000):
    y_train,x_test=y_train.to(device),x_test.to(device)
    out=model(y_train)
    loss=criterion(out,x_test)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%10==0:
        print('次数',i,'loss',loss.item())


test=y_test.unsqueeze(1).unsqueeze(1).to(device)
print(test.shape)
model.eval()  # 切换到评估模式
with torch.no_grad():
    pred=model(test)
    print(pred)

