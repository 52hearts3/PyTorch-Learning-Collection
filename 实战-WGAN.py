import torch
from torch import nn,optim,autograd
import random
import numpy as np

h_dim=400
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.net=nn.Sequential(
            # x [b,2]==>[b,2]  输出为2是为了可视化，输入的2可以随意定
            nn.Linear(2,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,2)
        )

    def forward(self,x):
        output=self.net(x)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.net=nn.Sequential(
            nn.Linear(2,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,1),
            nn.Sigmoid()  #代表概率
        )

    def forward(self,x):
        output=self.net(x)
        return output.view(-1)


#生成数据集
batch_size=512
def data_generator():
    # 8个高斯分布
    scale=2
    centers=[
        (1,0),
        (-1,0),
        (0,1),
        (0,-1),
        (1.0/np.sqrt(2),1.0/np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2))
    ]
    centers=[(scale*x,scale*y) for x,y in centers]

    while True:
        dataset=[]
        for i in range(batch_size):
            point=np.random.randn(2)*0.02
            center=random.choice(centers)
            # N(0,1) + center x1或x2
            point[0]+=center[0]
            point[1]+=center[1]
            dataset.append(point)
        dataset=np.array(dataset).astype(np.float32)
        dataset=dataset/1.414
        yield dataset  #每一次生成一个dataset就会返回这个dataset，并保存dataset的状态继续循环
def gradient_penalty(D,x_real,x_fake):
    #[b,1]
    t=torch.rand(batch_size,1).cuda()
    #[b,1]==>[b,2]
    t=t.expand_as(x_real)

    mid=t*x_real+(1-t)*x_fake
    mid.requires_grad_()
    pred=D(mid)
    grads=autograd.grad(outputs=pred,inputs=mid,
                        grad_outputs=torch.ones_like(pred),
                        create_graph=True,retain_graph=True,only_inputs=True)[0] #对mid求导
    gp=torch.pow(grads.norm(2,dim=1)-1,2).mean() #grads.norm(2,dim=1)求l2范数
    return gp

def main():
    torch.manual_seed(32)
    np.random.seed(32)
    data_iter=data_generator()
    x=next(data_iter)
    # [b,2]
    #print(x.shape)

    G=Generator().cuda()
    D=Discriminator().cuda()
    optimizer_G=optim.Adam(G.parameters(),lr=5e-4,betas=(0.5,0.9))
    optimizer_D=optim.Adam(D.parameters(),lr=5e-4,betas=(0.5,0.9))
    for epoch in range(1000):
        # 1 train Discriminator first
        for _ in range(5):
            # 1 在真实数据上训练
            x_real=next(data_iter)
            x_real=torch.tensor(x_real,dtype=torch.float32).cuda()
            pred_real=D(x_real)
            # 最大化 pred_real，最小化 loss_real
            loss_real = -pred_real.mean()
            # 2 在假数据上训练
            z=torch.randn(batch_size,2).cuda()
            x_fake=G(z).detach() #防止计算G的梯度
            pred_fake=D(x_fake)
            loss_fake=pred_fake.mean()

            #loss总和+梯度惩罚
            gp=gradient_penalty(D,x_real,x_fake.detach())
            loss_D=loss_fake+loss_real+0.2*gp
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()


        # 2 train Generator
        z = torch.randn(batch_size, 2).cuda()
        x_fake=G(z)
        pred_fake=D(x_fake)
        #最大化 loss_G
        loss_G = -pred_fake.mean()

        #optimize
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if epoch % 100 == 0:
            print('loss_D',loss_D.item(),'loss_G',loss_G.item())




if __name__=='__main__':
    main()