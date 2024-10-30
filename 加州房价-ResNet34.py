from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch import nn,optim
from sklearn.metrics import r2_score
import numpy as np

data_in=fetch_california_housing()
data=data_in.data
target=data_in.target
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.3,random_state=3)
# print(x_train.shape)

x_train=torch.tensor(x_train,dtype=torch.float32).unsqueeze(1).unsqueeze(1)
x_test=torch.tensor(x_test,dtype=torch.float32).unsqueeze(1).unsqueeze(1)
y_train=torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
y_test=torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)
#print(x_train.shape)

train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

        self.extra=nn.Sequential(
            nn.Conv2d(ch_in,ch_in,stride=stride,kernel_size=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU()
        )  #ch_in=ch_in

        if ch_in != ch_out:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride,padding=0),
                nn.BatchNorm2d(ch_out),
                nn.ReLU()
            )

    def forward(self,x):
        out=self.conv(x)
        #print(out.size())
        #print(self.extra(x).size())
        x=out+self.extra(x)
        return x

model=ResBlk(1,1,2)
test=torch.randn(32,1,1,8)
model(test)

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        )

        self.blk1=self.make_layer(64,64,num_blocks=3)
        self.blk2=self.make_layer(64,128,num_blocks=2,stride=2)
        self.blk3=self.make_layer(128,256,num_blocks=6,stride=2)
        self.blk4=self.make_layer(256,512,num_blocks=3,stride=2)

        self.linear=nn.Linear(512,1)
    def make_layer(self,ch_in,ch_out,num_blocks,stride=1):
        layer=[]
        layer.append(ResBlk(ch_in,ch_out,stride=stride))
        for _ in range(1,num_blocks):
            layer.append(ResBlk(ch_out,ch_out))
        return nn.Sequential(*layer)

    def forward(self,x):
        x=self.conv(x)
        x=self.blk1(x)
        x=self.blk2(x)
        x=self.blk3(x)
        x=self.blk4(x)
        x=nn.functional.adaptive_avg_pool2d(x,(1,1))
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=ResNet34().to(device)
x_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=2)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=2)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=1e-4)


def main():
    for epoch in range(1000):
        model.train()
        for x,y in x_loader:
            x_,y_=x.to(device),y.to(device)
            y_hat=model(x_)
            loss=criterion(y_hat,y_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for x,label in test_loader:
                    x, label = x.to(device), label.to(device)
                    pred = model(x)
                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(label.cpu().numpy())
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                score = r2_score(all_labels, all_preds)
                print(f'R² Score: {score}')
        print('epoch:', epoch, 'loss:', loss.item())

import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.freeze_support()
    main()




