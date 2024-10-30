import torch
from torch import nn,optim,autograd
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
# def weights_init(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         nn.init.constant_(m.bias, 0)
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.net=nn.Sequential(
            nn.Linear(100,512*7*7),
            nn.ReLU(),
            nn.Unflatten(1,(512,7,7)), #[b,512*7*7]==>[b,512,7,7]
            nn.ReLU(),
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
        # self.net.apply(weights_init)

    def forward(self,x):
        x=self.net(x)
        #print(x.shape)
        return x

# test=torch.randn(32,100)
# model=Generator()
#print(model(test).shape)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.net=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(512),
        )

        self.linear=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*14*14,1)
        )
        # self.net.apply(weights_init)
        # self.linear.apply(weights_init)

    def forward(self,x):
        x=self.net(x)
        #print(x.shape)  #[32, 512, 14, 14]
        x=self.linear(x)
        x = x.view(-1)
        return x

# test=torch.randn(32,3,224,224)
# model=Discriminator()
#print(model(test).shape)

from PIL import Image
# def convert_to_rgb(image):
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
#     return image


tf=transforms.Compose([
    #transforms.Lambda(convert_to_rgb),
    transforms.Resize((224,224)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data=datasets.CIFAR10(root=r'D:\game\pytorch\简单分类问题\data', train=True, transform=tf, download=True)

loader=DataLoader(data,batch_size=32,shuffle=True,num_workers=2)

Generator=Generator().cuda()
Discriminator=Discriminator().cuda()
optimizer_G=optim.Adam(Generator.parameters(),lr=1e-4, betas=(0.5, 0.999))
optimizer_D=optim.Adam(Discriminator.parameters(),lr=1e-4, betas=(0.5, 0.999))

import matplotlib.pyplot as plt

def show_generated_images(epoch, generator, num_images=5):
    z = torch.randn(num_images, 100).cuda()
    fake_images = generator(z).cpu().detach()
    fake_images = (fake_images + 1) / 2   # 将图像从 [-1, 1] 转换到 [0, 1]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(fake_images[i].permute(1, 2, 0).numpy())
        axes[i].axis('off')
    plt.suptitle(f'Epoch {epoch + 1}')
    plt.show()

def gradient_penalty(D,x_real,x_fake,batch_size):
    #[b,1,1,1]
    t=torch.rand(batch_size,1,1,1).cuda()
    #[b,3,224,224]
    t=t.expand_as(x_real)

    mid=t*x_real+((1-t)*x_fake)
    mid.requires_grad_()
    pred=D(mid)
    grads=autograd.grad(outputs=pred,inputs=mid,
                        grad_outputs=torch.ones_like(pred),
                        create_graph=True,retain_graph=True,only_inputs=True)[0] #对mid求导
    gradient=grads.view(grads.size(0),-1)
    gp=((gradient.norm(2, dim=1) - 1) ** 2).mean()  #grads.norm(2,dim=1)求l2范数
    return gp

def main():
    for epoch in range(1000):
        for x,_ in loader:
            for p in Discriminator.parameters():
                p.requires_grad = True
            #训练判别器
            x_hat=x.cuda()
            batch_size=x_hat.size(0)
            #print(batch_size)
            for _ in range(5):
                z = torch.randn(batch_size, 100).cuda()
                fake_x = Generator(z).detach()
                real_loss = -torch.mean(Discriminator(x_hat))
                fake_loss = torch.mean(Discriminator(fake_x))

                c_loss = real_loss + fake_loss + 10*gradient_penalty(Discriminator,x_real=x_hat,x_fake=fake_x.detach(),batch_size=batch_size)

                optimizer_D.zero_grad()
                c_loss.backward()
                # 梯度裁剪
                # torch.nn.utils.clip_grad_norm_(Discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()
                #print(f'Discriminator Loss: {c_loss.item()}')


            for p in Discriminator.parameters():
                p.requires_grad = False
            # 训练生成器
            z = torch.randn(batch_size, 100).cuda()
            fake_x = Generator(z)
            g_loss = -torch.mean(Discriminator(fake_x))

            optimizer_G.zero_grad()
            g_loss.backward()
            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(Generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            #print(f'Generator Loss: {g_loss.item()}')

        print(f'Epoch [{epoch + 1}/1000], c_loss: {c_loss.item()}, g_loss: {g_loss.item()}')
        if epoch % 10 == 0:
            show_generated_images(epoch, Generator)

import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.freeze_support()
    main()