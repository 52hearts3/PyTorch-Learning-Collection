import torch
from torch import nn,optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Attention_block(nn.Module):
    def __init__(self,ch_in):
        super(Attention_block,self).__init__()

        self.ch_in=ch_in
        self.to_qkv=nn.Conv2d(ch_in,ch_in*3,kernel_size=1,stride=1,padding=0)
        self.to_out=nn.Conv2d(ch_in,ch_in,kernel_size=1)

        self.norm=nn.Sequential(
            nn.BatchNorm2d(ch_in)
        )

    def forward(self,x):
        b,ch,h,w=x.size()
        x_norm=self.norm(x)
        x_qkv=self.to_qkv(x_norm)
        q,k,v=torch.split(x_qkv,self.ch_in,dim=1)
        q=q.permute(0,2,3,1).view(b,h*w,ch)
        k=k.view(b,ch,h*w)
        v=v.permute(0,2,3,1).view(b,h*w,ch)

        dot=torch.bmm(q,k)*(ch**-0.5)
        attention=torch.softmax(dot,dim=-1)
        out=torch.bmm(attention,v).view(b,h,w,ch).permute(0,3,1,2)
        return self.to_out(out)+x

test=torch.randn(12,3,32,32)
model=Attention_block(ch_in=3)
print(model(test).size())

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk,self).__init__()

        self.ch_in=ch_in

        self.conv=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

        self.attention=Attention_block(ch_out)
        self.shortcut=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self,x):
        out=self.conv(x)
        out=self.attention(out)
        x=out+self.shortcut(x)
        return x

test=torch.randn(12,3,32,32)
model=ResBlk(ch_in=3,ch_out=16)
print(model(test).size())
class up_sample(nn.Module):
    def __init__(self,ch_in):
        super(up_sample,self).__init__()

        self.up_sample=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=1,padding=1)
        )

    def forward(self,x):
        x=self.up_sample(x)
        return x

class down_sample(nn.Module):
    def __init__(self,ch_in):
        super(down_sample,self).__init__()

        self.down_sample=nn.Sequential(
            nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.down_sample(x)
        return x

class Unet(nn.Module):
    def __init__(self,ch_in,dim):
        super(Unet,self).__init__()

        self.conv1=nn.Conv2d(ch_in,dim,kernel_size=3,stride=1,padding=1)

        #下采样 每一次加入两个短接层
        down=[]
        for i in range(4):
            for _ in range(2):
                down.append(ResBlk(ch_in=dim,ch_out=dim))
            down.append(down_sample(ch_in=dim))

        self.down_sample=nn.Sequential(*down)

        self.mid_layer=nn.Sequential(
            ResBlk(ch_in=dim,ch_out=dim),
            ResBlk(ch_in=dim,ch_out=dim)
        )

        #上采样 每一次加入两个短接层
        up=[]
        for i in range(4):
            for _ in range(2):
                up.append(ResBlk(ch_in=dim,ch_out=dim))
            up.append(up_sample(ch_in=dim))

        self.up_sample=nn.Sequential(*up)

        self.change_ch = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, ch_in, 3, padding=1, stride=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.conv1(x)
        #下采样
        down_output=[]
        for layer in self.down_sample:
            x=layer(x)
            down_output.append(x)

        x=self.mid_layer(x)
        #上采样
        for layer in self.up_sample:
            skip_connection = down_output.pop()
            if skip_connection.size() != x.size():
                skip_connection = torch.nn.functional.interpolate(skip_connection, size=x.size()[2:])
            x = torch.cat((x, skip_connection), dim=1)  # [b,ch,x,x]==>[b,2ch,x,x]
            x = self.change_ch(x)
            # print('s',x.size())
            # print(x.size())
            x = layer(x)

        x=self.out_conv(x)
        return x

test=torch.randn(12,3,32,32)
model=Unet(ch_in=3,dim=64)
print(model(test).size())



#预训练uUet

def pretrain_Unet(Unet,data_loader,epochs=1,lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Unet.parameters(), lr=lr)

    for epoch in range(epochs):
        Unet.train()
        running_loss = 0.0
        for images, labels in data_loader:
            optimizer.zero_grad()
            images,labels=images.to(device),labels.to(device)
            outputs = Unet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}')


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



