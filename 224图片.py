import torch
from torch import nn,optim,autograd
import random
import numpy as np
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

h_dim = 400
batch_size = 128

#生成器需要将随机噪声转换为 224x224 的图片。可以使用反卷积层（也称为转置卷积层）来逐步增加图片的尺寸。
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 512 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


#判别器需要将 224x224 的图片转换为一个概率值。可以使用卷积层来逐步减少图片的尺寸，并使用全连接层进行分类。
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            #nn.LeakyReLU 旨在解决 ReLU 可能出现的“神经元死亡”问题
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 1),  # 确保输入尺寸匹配
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_dataset = datasets.CIFAR10(root=r'D:\game\pytorch\简单分类问题\data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
G = Generator().cuda()
D = Discriminator().cuda()
optimizer_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# 训练循环
for epoch in range(100):
    for i, (images, _) in enumerate(train_loader):
        images = images.cuda()
        batch_size = images.size(0)

        # 训练判别器
        z = torch.randn(batch_size, 100).cuda()
        fake_images = G(z)
        real_labels = torch.ones(batch_size).cuda().unsqueeze(1)
        fake_labels = torch.zeros(batch_size).cuda().unsqueeze(1)

        outputs = D(images)
        d_loss_real = nn.BCELoss()(outputs, real_labels)
        real_score = outputs

        outputs = D(fake_images.detach())
        d_loss_fake = nn.BCELoss()(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        outputs = D(fake_images)
        g_loss = nn.BCELoss()(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch+1}/100], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
