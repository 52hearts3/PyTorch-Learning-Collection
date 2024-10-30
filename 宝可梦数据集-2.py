import torch
from torch import nn,optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
class V_AutoEncoder(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(V_AutoEncoder,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,3,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(3)
        )

        self.encoder=nn.Sequential(
            nn.Linear(ch_in,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.ReLU()
        )

        self.decoder=nn.Sequential(
            nn.Linear(5,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,ch_out),
            nn.Sigmoid()
        )
    def test_dim(self,x):
        x=self.conv(x)
        return x


    def forward(self,x):
        x=self.conv(x)
        #print(x.shape)
        x=x.view(x.size(0),-1)
        h_=self.encoder(x)
        mu,sigma=h_.chunk(2,dim=1)
        h=mu+sigma*torch.randn_like(sigma)
        new=self.decoder(h)
        h_hat=new.view(x.size(0),3,224,224)
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) + torch.pow(sigma, 2) - torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (x.size(0) * 224 * 224 * 3)
        return h_hat,kld

test=torch.randn(32,3,224,224)
model_test=V_AutoEncoder(1,2)
ch_in=model_test.test_dim(test).view(test.size(0),-1).size(1)
ch_out=test.view(test.size(0),-1).size(1)
# model=V_AutoEncoder(ch_in,ch_out)
# print(model(test)[0].size())

tf=transforms.Compose([
    transforms.Resize((int(224*1.5),int(224*1.5))),
    transforms.RandomRotation(15),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
data=torchvision.datasets.ImageFolder(root=r'D:\game\pytorch\自定义数据集实战\pokeman',transform=tf)
loader=DataLoader(data,batch_size=32,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=V_AutoEncoder(ch_in,ch_out).to(device)
optimizer=optim.Adam(model.parameters(),lr=1e-4)
criterion=nn.MSELoss()

def test(sample_size,model):
    model.eval()
    with torch.no_grad():
        x=torch.randn(sample_size,5).to(device)
        sample=model.decoder(x)
        samples = sample.view(sample_size, 3, 224, 224)
        for i in range(sample_size):
            img=samples[i].cpu().numpy().transpose(1,2,0)
            plt.imshow(img)
            plt.show()



def main():
    model.train()
    for epoch in range(500):
        for x,_ in loader:
            x=x.to(device)
            x_hat,kld=model(x)

            loss=criterion(x_hat,x)+1.*kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10==0:
            print('loss:',loss.item(),'kld:',kld.item())
    with torch.no_grad():
        test(10,model)

if __name__ == '__main__':
    main()