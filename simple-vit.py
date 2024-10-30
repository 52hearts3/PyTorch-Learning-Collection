import torch
from torch import nn,optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
def position_embed_sin_cos_2d(h,w,dim,temperature=1000): #加上位置嵌入，保留图像块的位置信息。
    y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    # indexing="ij" 指定生成的网格使用矩阵索引方式（即行列索引），这意味着 y 表示行索引，x 表示列索引。
    # 对于 ( h = 3 ) 和 ( w = 3 ) 的情况：
    # y = tensor([[0, 0, 0],
    #             [1, 1, 1],
    #             [2, 2, 2]])
    #
    # x = tensor([[0, 1, 2],
    #             [0, 1, 2],
    #             [0, 1, 2]])
    # y 的每一行都是相同的行索引：表示在 y 张量中，每一行的值都是相同的，且等于该行的索引。
    # x 的每一列都是相同的列索引：表示在 x 张量中，每一列的值都是相同的，且等于该列的索引
    omega=torch.arange(dim//4)/(dim//4-1)
    omega=1.0/(temperature**omega)
    # 用于计算频率向量 omega，它在生成二维正弦余弦位置编码时起到缩放作用
    y = y.flatten().view(-1,1) * omega.view(1,-1) #[h,w]==>[h*w]==>[h*w,1]*[1,dim//4]==>[h*w,dim//4]
    x = x.flatten().view(-1,1) * omega.view(1, -1)
    pe=torch.cat((x.sin(),x.cos(),y.sin(),y.cos()),dim=1) #[h*w,dim//4]==>[h*w,dim]
    pe=pe.to(torch.float32)
    return pe

test=position_embed_sin_cos_2d(h=32,w=32,dim=8)
print(test.size())

class Feedforward(nn.Module): #前馈神经网络,这个网络通常用于Transformer模型中的位置编码或其他需要非线性变换的地方
    def __init__(self,dim,dim_hidden):
        super(Feedforward,self).__init__()

        self.net=nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden,dim)
        )

    def forward(self,x):
        x=self.net(x)
        return x

class Attention_block(nn.Module):  # 多头注意力机制
    def __init__(self,dim,heads=8,dim_head=64):
        super(Attention_block,self).__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self,x):
        x=self.norm(x)
        b, n, chd = x.size()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, -1).permute(0, 2, 1, 3).contiguous(), qkv)
        # ==>[b,n,h,d]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # torch.matmul 是矩阵乘法操作，这里计算的是 q 和转置后的 k 之间的矩阵乘法。
        # [b,n,h,d]*[b,n,d,h]==>[b,n,h,h]
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        # [b,n,h,h]*[b,n,h,d]==>[b,n,h,d]
        #print(out.size())
        out=out.permute(0,2,1,3).contiguous().view(b, n, -1)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self,dim,dim_hidden,depth=5):
        super(Transformer,self).__init__()

        self.norm=nn.LayerNorm(dim)

        layer=[]
        for _ in range(depth):
            layer.append(Attention_block(dim=dim,heads=8))
            layer.append(Feedforward(dim=dim,dim_hidden=dim_hidden))

        self.layers=nn.Sequential(*layer)

    def forward(self,x):
        for layer in self.layers:
            if isinstance(layer,Attention_block):
                x=layer(x) + x
            if isinstance(layer,Feedforward):
                x=layer(x) + x
        return self.norm(x)

class Simple_ViT(nn.Module):
    def __init__(self,image_h,image_w,dim,patch_h,patch_w,num_classes,ch_in=3):
        super(Simple_ViT,self).__init__()

        self.patch_h=patch_h
        self.patch_w=patch_w

        patch_dim=ch_in*patch_h*patch_w

        self.to_patch_embedding=nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim,dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding=position_embed_sin_cos_2d(
            h=image_h/patch_h, #选这个参数是因为 x=x.view(b,-1,p1*p2*c)==>[b,(h*w)/(p1*p2),p1*p2*c]
            w=image_w/patch_w,
            dim=dim
        )

        self.transformer=Transformer(
            dim=dim,
            dim_hidden=64,
        )

        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)  #分类

    def forward(self,x):
        device=x.device
        b,c,h,w=x.size()
        p1,p2=self.patch_h,self.patch_w
        x=x.view( b , c, h//p1 , p1 , w//p2 , p2 ).permute(0, 2, 4, 3, 5, 1).contiguous()
        # h // p1：每个patch的高度块数  w // p2：每个patch的宽度块数
        # 将输入图像分割成多个小块（patches），每个小块的形状为 [p1, p2, c]
        x=x.view(b,-1,p1*p2*c)
        # 通过这些步骤，输入图像被分割成多个小块，每个小块被展平并重新排列成适合Vision Transformer处理的形状
        # 重新排列并展平这些小块，得到形状为 [b, -1, p1 * p2 * c] 的张量。
        x=self.to_patch_embedding(x)
        # 通过 to_patch_embedding 层将图像块转换为嵌入向量。
        # 注意，nn.Linear其实也可以处理非二维张量，这里[b, num_patches, patch_dim]==>[b,num_patches,dim]
        # print(x.size()) #torch.Size([12, 16, 192])
        x = x + self.pos_embedding.to(device,dtype=x.dtype)
        # 加上位置嵌入，保留图像块的位置信息

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)

model=Simple_ViT(ch_in=3,image_h=32,image_w=32,dim=512,patch_h=8,patch_w=8,num_classes=10)
test=torch.randn(12,3,32,32)
print(model(test).size())

tf=transforms.Compose([
    #transforms.Lambda(convert_to_rgb),
    transforms.Resize((32,32)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_train=datasets.CIFAR10(root=r'D:\game\pytorch\简单分类问题\data', train=True, transform=tf, download=True)
data_test=datasets.CIFAR10(root=r'D:\game\pytorch\简单分类问题\data', train=False, transform=tf, download=True)
train_loader=DataLoader(data_train,batch_size=32,shuffle=True)
test_loader=DataLoader(data_test,batch_size=32,shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Simple_ViT(ch_in=3,image_h=32,image_w=32,dim=512,patch_h=8,patch_w=8,num_classes=10).to(device)
criteon=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)

model.train()
for epoch in range(1000):
    for batch_idx,(x,label) in enumerate(train_loader):
        x,label=x.to(device),label.to(device)
        logits=model(x)
        loss=criteon(logits,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(epoch,loss.item())
    #test
    model.eval()
    with torch.no_grad():  #对于分类问题，神经网络的预测结果并不是标签，而是不同标签的概率分布
        total_correct = 0
        total_sum = 0
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred, label).float().sum().item()  # 使用item转换为numpy
            total_sum += x.size(0)
        acc = total_correct / total_sum
        print(acc)