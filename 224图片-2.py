import torch
from torch import nn,optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.net=nn.Sequential(
            nn.Linear(100,512*7*7),
            nn.ReLU(True),
            nn.Unflatten(1,(512,7,7)),
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self,x):
        x=self.net(x)
        return x

test=torch.randn(512,100)
model=Generator()
print(model(test).shape)  #[512, 3, 224, 224]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.net=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU 旨在解决 ReLU 可能出现的“神经元死亡”问题
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.linear=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 1),  # 确保输入尺寸匹配
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.net(x)
        x=self.linear(x)
        return x

test=torch.randn(512,3,224,224)
model=Discriminator()
print(model(test).shape)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root=r'D:\game\pytorch\简单分类问题\data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

Generator=Generator().cuda()
Discriminator=Discriminator().cuda()
optimizer_G = optim.Adam(Generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(Discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

def show_generated_images(epoch, generator, num_images=5):
    z = torch.randn(num_images, 100).cuda()
    fake_images = generator(z).cpu().detach()
    fake_images = (fake_images + 1) / 2  # 将图像从 [-1, 1] 转换到 [0, 1]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(fake_images[i].permute(1, 2, 0).numpy())
        axes[i].axis('off')
    plt.suptitle(f'Epoch {epoch + 1}')
    plt.show()


for epoch in range(100):
    for x,_ in train_loader:
        x_hat=x.cuda()
        batch_size=x_hat.size(0) #  32

        #训练判别器
        z=torch.randn(batch_size,100).cuda()
        fake_x=Generator(z)
        real_labels = torch.ones(batch_size).cuda().unsqueeze(1)*0.9
        fake_labels = torch.zeros(batch_size).cuda().unsqueeze(1)+0.1

        outputs = Discriminator(x_hat)
        d_loss_real = nn.BCELoss()(outputs, real_labels)
        real_score = outputs

        outputs = Discriminator(fake_x.detach())
        d_loss_fake = nn.BCELoss()(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器

        outputs = Discriminator(fake_x)
        g_loss = nn.BCELoss()(outputs, real_labels)
        optimizer_G.zero_grad()
        g_loss.backward()  # 保留计算图
        optimizer_G.step()

    print(f'Epoch [{epoch + 1}/100], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
    show_generated_images(epoch, Generator)