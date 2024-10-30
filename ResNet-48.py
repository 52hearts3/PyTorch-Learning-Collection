import torch
from torch import nn,optim
from torchvision import transforms
import torchvision
from torch.utils.data import random_split,DataLoader
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
            nn.Conv2d(ch_in,ch_in,kernel_size=1,stride=stride,padding=0),
            nn.BatchNorm2d(ch_in)
        )

        if ch_in !=ch_out:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,stride=stride,kernel_size=1,padding=0),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        out=self.conv(x)
        #print(out.shape)
        out=out+self.extra(x)
        #print(self.extra(x).shape)
        return out

# test=torch.randn(32,3,224,224)
# model=ResBlk(3,64)
# print(model(test).shape)

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.blk1=self._make_layer(64,64,num_blocks=3)
        self.blk2=self._make_layer(64,128,num_blocks=2,stride=2)
        self.blk3=self._make_layer(128,256,num_blocks=6,stride=2)
        self.blk4=self._make_layer(256,512,num_blocks=3,stride=2)

        self.linear=nn.Linear(512,5)

    def _make_layer(self, ch_in, ch_out, num_blocks, stride=1):
        layers = []
        layers.append(ResBlk(ch_in, ch_out, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlk(ch_out, ch_out))
        return nn.Sequential(*layers)

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

# test=torch.randn(32,3,224,224)
# model=ResNet34()
# print(model(test).shape)

tf = transforms.Compose([
        transforms.Resize((int(224*1.25),int(224*1.25))),  # resize的大一点以便数据增强
        transforms.RandomRotation(15),  # 随机旋转15度
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

data=torchvision.datasets.ImageFolder(root=r'D:\game\pytorch\自定义数据集实战\pokeman',transform=tf)
train_size=int(0.7*len(data))
test_size=len(data)-train_size
train_dataset,test_dataset=random_split(data,[train_size,test_size])
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=2)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    model=ResNet34().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-4)

    model.train()
    for epoch in range(1000):
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            out=model(x)
            loss=criterion(out,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_sum = 0
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)
                # [b,10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()  # 使用item转换为numpy
                total_sum += x.size(0)
            acc = total_correct / total_sum
            print(acc)

import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.freeze_support()
    main()





