import torch
from torch import nn,optim

def position_embed_sin_cos_2d(h,w,dim,t,temperature=1000): #加上位置嵌入，保留图像块的位置信息。
    device=t.device
    y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    omega=torch.arange(dim//4)/(dim//4-1)
    omega=1.0/(temperature**omega)
    y=y.flatten().view(-1,1)*omega.view(1,-1) #[h*w,dim//4]
    x=x.flatten().view(-1,1)*omega.view(1,-1)
    pe=torch.cat((x.sin(),x.cos(),y.sin(),y.cos()),dim=1)
    pe=pe.to(torch.float32)
    return pe

#可以减小模型复杂度，避免过拟合
class LayerNorm(nn.Module): #实现了一个无偏置的层归一化（LayerNorm），并使用了一个称为“单位偏移技巧”（unit offset trick）的方法
    def __init__(self,dim):
        super(LayerNorm,self).__init__()

        self.layer_norm=nn.LayerNorm(dim,elementwise_affine=False)  #不使用可学习的缩放和平移参数。
        self.gamma = nn.Parameter(torch.zeros(dim))
    def forward(self,x):
        norm=self.layer_norm(x)
        return norm*(self.gamma+1)

class MLP(nn.Module):  #多层感知机
    def __init__(self,dim,hidden_dim):
        super(MLP,self).__init__()

        self.net=nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(0.2)
        )

    def forward(self,x):
        x=self.net(x)
        return x


class Attention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.,cross_attend=False,reuse_attention=False):
        super(Attention,self).__init__()
        # heads-->注意力头的数量，dim_head-->每个注意力头的维度，cross_attend-->是否使用交叉注意力
        inner_dim=dim_head*heads

        self.scale=dim_head**-0.5 #缩放因子

        self.heads = heads
        self.reuse_attention = reuse_attention
        self.cross_attend = cross_attend

        self.norm=LayerNorm(dim) if  reuse_attention==False else nn.Identity()
        self.norm_context=LayerNorm(dim) if cross_attend==True else nn.Identity()

        self.attend=nn.Softmax(-1)
        self.dropout=nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False) if  reuse_attention==False else None
        self.to_k = nn.Linear(dim, inner_dim, bias=False) if  reuse_attention==False else None
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self,x,context=None,return_qk_sim=None,qk_sim=None):
        # context为上下文张量
        # 通过传入 context，ViT 模型可以利用额外的信息源，从而增强其特征表示能力和任务表现。
        x=self.norm(x)

        if self.cross_attend:
            context = self.norm_context(context)
        else:
            context = x
        v=self.to_v(context)

        b,n,hd=v.size()
        v=v.view(b,n,self.heads,-1).permute(0,2,1,3).contiguous()

        #计算q和k
        if not self.reuse_attention:
            q,k=self.to_q(x),self.to_k(x)
            q = q.view(b,n, self.heads, -1).permute(0, 2, 1, 3).contiguous()
            k = k.view(b,n, self.heads, -1).permute(0, 2, 1, 3).contiguous()
            q=q*self.scale
            qk_sim = torch.matmul(q, k.transpose(-2, -1))  #计算q和k的点积相似度

        attend=self.attend(qk_sim)
        attention=self.dropout(attend)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        out=self.to_out(out)
        if return_qk_sim==True:
            return out,qk_sim
        else:
            return out

test=torch.randn(3,12,8*8)
context=torch.randn_like(test)
model=Attention(dim=64,dropout=0.2,cross_attend=True,reuse_attention=False)
print(model(test,context).size())

class look_vit(nn.Module):
    def __init__(self,
        dim,
        image_size,
        num_classes,
        depth = 3,
        patch_size = 16,  #图像补丁的大小，默认值为 16。表示将图像分割成 16x16 的小块。
        heads = 8,
        mlp_factor = 4,
        dim_head = 64,
        highres_patch_size = 8, #高分辨率补丁的大小，默认值为 12。表示高分辨率图像分割成 12x12 的小块。
        highres_mlp_factor = 4, #高分辨率 MLP 层的扩展因子，默认值为 4。表示高分辨率 MLP 层中隐藏层的维度是输入维度的多少倍。
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        patch_conv_kernel_size = 7,
        dropout = 0.1,
        channels = 3
    ):
        super(look_vit,self).__init__()

        self.highres_patch_size=highres_patch_size
        self.factor=mlp_factor
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size

        kernel_size = patch_conv_kernel_size
        patch_dim=(highres_patch_size*highres_patch_size)*channels

        self.to_patches=nn.Sequential(
            #view一次  [b ,c, (h*p1), (w*p2)] -> [b, (p1*p2*c), h, w]
            nn.Conv2d(patch_dim,dim,kernel_size=kernel_size,padding=kernel_size//2),
            #view一次  [b c h w] -> [b h w c]
        )

        #位置嵌入
        num_patches=(image_size//highres_patch_size)**2
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))

        #look-vit定义
        layers=nn.ModuleList([])
        for _ in range(depth):
            layers.append(
                nn.ModuleList([
                Attention(dim=dim,heads=heads,dim_head=dim_head,dropout=dropout),
                MLP(dim=dim,hidden_dim=dim*mlp_factor),
                Attention(dim=dim, dim_head=cross_attn_dim_head, heads=cross_attn_heads, dropout=dropout,cross_attend=True),
                Attention(dim=dim, dim_head=cross_attn_dim_head, heads=cross_attn_heads, dropout=dropout,cross_attend=True, reuse_attention=True),
                LayerNorm(dim),
                MLP(dim=dim,hidden_dim=dim*highres_mlp_factor)])
            )

        self.layers = layers

        self.norm = LayerNorm(dim)
        self.highres_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, num_classes, bias=False)

    def forward(self,x):
        b,c,h,w=x.size()
        p1=p2=self.highres_patch_size
        x = x.view(b, c, h // p1, p1, w // p2, p2).permute(0, 2, 4, 3, 5, 1).contiguous()
        x=x.view(b,(h//p1)*(w//p2),p1*p2*c).view(b,h//p1,w//p2,p1*p2*c)
        x = x.permute(0, 3, 1, 2).contiguous()  #[b,p1*p2*c,h//p1,w//p2]

        highres_tokens = self.to_patches(x)
        highres_tokens=highres_tokens.permute(0,2,3,1)
        #print(highres_tokens.device)
        highres_tokens=LayerNorm(dim=self.dim)(highres_tokens)

        size = highres_tokens.shape[-2]
        h_,w_=self.image_size//p1,self.image_size//p2
        #print('h',h_,'w',w_)
        pos_emb = position_embed_sin_cos_2d(h=h_,w=w_,dim=self.dim,t=x)
        print(pos_emb.size())
        print(highres_tokens.size())
        highres_tokens = highres_tokens + pos_emb.view(size,-1,self.dim)
        print(highres_tokens.size())  #torch.Size([12, 4, 4, 64])

        highres_tokens_2=highres_tokens.permute(0,3,1,2)
        tokens=nn.functional.interpolate(  #F.interpolate：PyTorch 中的插值函数，用于对输入张量进行上采样或下采样。
            highres_tokens_2,
            w//self.patch_size,  #img.shape[-1] // self.patch_size：计算插值后的目标大小
            mode='bilinear'
        ) #将高分辨率补丁插值到与主补丁相同的大小，以便在后续的 Transformer 层中进行处理。
        print(tokens.size())

        b_,c_,_,_=tokens.size()
        b__,_,_,c__=highres_tokens.size()
        tokens=tokens.permute(0,2,3,1).view(b_,-1,c_) #[12, 64, 32, 32]==>[12, 1024, 64]
        print('gg',highres_tokens.size())
        print(highres_tokens.permute(0, 2, 3, 1).size())
        highres_tokens=highres_tokens.permute(0,2,3,1).contiguous().view(b__,-1,c__)
        print(tokens.size())
        print('gg',highres_tokens.size())

        # look-vit向前传播和注意力机制
        for attn, mlp, lookup_cross_attn, highres_attn, highres_norm, highres_mlp in self.layers:
            lookup_out, qk_sim = lookup_cross_attn(tokens, highres_tokens, return_qk_sim=True)
            tokens = lookup_out + tokens #残差连接：将交叉注意力输出与原始主补丁相加。

            tokens = attn(tokens) + tokens
            tokens = mlp(tokens) + tokens

            b,n,i,j=qk_sim.size()
            qk_sim=qk_sim.view(b,n,j,i)  #将注意力矩阵转置，以便进行反向交叉注意力计算。
            highres_tokens = highres_attn(highres_tokens, tokens, qk_sim=qk_sim) + highres_tokens
            highres_tokens = highres_norm(highres_tokens)

            highres_tokens = highres_mlp(highres_tokens) + highres_tokens

        tokens = self.norm(tokens)
        highres_tokens = self.highres_norm(highres_tokens)
        tokens = torch.mean(tokens, dim=1)
        highres_tokens = torch.mean(highres_tokens, dim=1)
        return self.to_logits(tokens+highres_tokens)


test=torch.randn(12,3,32,32)
model=look_vit(dim=64,image_size=32,num_classes=10,patch_size=8)
print(model(test).size())