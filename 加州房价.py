from sklearn.datasets import fetch_california_housing
import torch
from torch import nn,optim
from torch.utils.data import DataLoader,TensorDataset
data_in=fetch_california_housing()
data=data_in.data
target=data_in.target

#考虑到时间，不能随机分训练集，测试集
L_train=int(data.shape[0]*0.8)
x_train=data[:L_train]
y_train=target[:L_train]
print(x_train.shape)

x_train=torch.tensor(x_train,dtype=torch.float32).unsqueeze(1)
x_test=torch.tensor(data[1:L_train+1],dtype=torch.float32).unsqueeze(1)
y_train=torch.tensor(y_train,dtype=torch.float32)
y_test=torch.tensor(target[1:L_train+1],dtype=torch.float32).unsqueeze(1).unsqueeze(1)
print(x_train.shape)
print(y_train.shape)



#我们的输入数据为3维，[x,y,z]
# x是时间步，也就是每个序列的长度
# y是序列个数，也就是我们希望同时处理多少个序列
# z是输入数据的维度，也就是对于每个时间序列，每一天的数据维度
# 对于本问题，x_train的shape为(16512, 8)，即x是16512，z是8

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(LSTM,self).__init__()
        self.LSTM=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
        # input_size可以理解为每天的数据维度，我们这里的特征为8
        self.linear=nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,1)
        )

    def forward(self,x):
        out,(h,c)=self.LSTM(x)
        seq_len,batch_size,hid_dim=out.shape
        out=out.view(-1,hid_dim)
        out=self.linear(out)
        out=out.view(seq_len,batch_size,-1)
        #print(out.shape)
        return out

#  test
x=torch.randn(32,1,8)
model=LSTM(8,10,1)
test=model(x)
print(test.shape)

model=LSTM(8,10,1)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=1e-4)
model.train()
for i in range(1000):
    out=model(x_train)
    loss=criterion(out,y_test)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%10==0:
        print('次数',i,'loss',loss.item())

# 假设要预测未来10个时间步的数据
future_steps = 10
# 准备输入数据，这里我们用最后一个训练样本作为起点
# 需要根据实际情况准备输入数据
input_data = x_train[-1].unsqueeze(0)  # 形状为 (1, 1, 8)

predictions = []

model.eval()  # 切换到评估模式
with torch.no_grad():
    for _ in range(future_steps):
        # 使用模型进行预测
        output = model(input_data)
        # 获取预测值
        predicted_value = output[-1, 0, 0].item()
        predictions.append(predicted_value)
        # 更新输入数据，将预测值添加到输入数据中
        new_input = torch.tensor(predicted_value, dtype=torch.float32).view(1, 1, 1)
        input_data = torch.cat((input_data[:, :, 1:], new_input), dim=2)
print(predictions)
