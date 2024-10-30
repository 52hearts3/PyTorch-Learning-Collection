import torch
from torch import nn,optim

def position_embed_sin_cos(w,h,dim,temperature=1000):
    y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')

    omega=torch.arange(dim//4)/(dim//4-1)
    omega=1/(omega**temperature)

    y=y.flatten().view(-1,1)*omega.view(1,-1)
    x=x.flatten().view(-1,1)*omega.view(1,-1)

    pe=torch.cat((x.sin(),x.cos(),y.sin(),y.cos()),dim=1)
    pe=pe.to(torch.float32)
    return pe

class LayerNorm(nn.Module):
    def __init__(self,dim):
        super(LayerNorm,self).__init__()

        self.norm=nn.LayerNorm(dim,elementwise_affine=False)
        self.gamma=nn.Parameter(torch.zeros(dim))

    def forward(self,x):
        x=self.norm(x)
        return x*(self.gamma+1)


class MLP(nn.Module):
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

        inner_dim=dim_head*heads
        self.scaler=dim_head**-0.5

        self.heads = heads
        self.reuse_attention = reuse_attention
        self.cross_attend = cross_attend

        self.norm=LayerNorm(dim) if reuse_attention==False else nn.Identity()
        self.norm_context=LayerNorm(dim) if cross_attend==True else nn.Identity()

        self.attend=nn.Softmax(-1)
        self.dropout=nn.Dropout(dropout)

        self.to_q=nn.Linear(dim,inner_dim,bias=False) if reuse_attention==False else None
        self.to_k=nn.Linear(dim,inner_dim,bias=False) if reuse_attention==False else None
        self.to_v=nn.Linear(dim,inner_dim,bias=False)

        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,dim,bias=False),
            nn.Dropout(dropout)
        )

    def forward(self,x,context=None,return_qk_sim=None,qk_sim=None):
        x=self.norm(x)

        if self.cross_attend:
            context=self.norm_context(context)
        else:
            context=x
        v=self.to_v(context)
        b,n,hd=v.size()
        v=v.view(b,n,self.heads,-1).permute(0,2,1,3).contiguous()

        if self.reuse_attention==False:
            q=self.to_q(x).view(b,n,self.heads,-1).permute(0,2,1,3).contiguous()
            k=self.to_k(x).view(b,n,self.heads,-1).permute(0,2,1,3).contiguous()
            q=q*self.scaler
            qk_sim=torch.matmul(q,k.transpose(-2,-1))

        attend=self.attend(qk_sim)
        attention=self.dropout(attend)
        out=torch.matmul(attention,v)
        out=out.permute(0,2,1,3).contiguous().view(b,n,-1)
        out=self.to_out(out)
        if return_qk_sim:
            return out,qk_sim
        else:
            return out

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

        self.highres_patch_size = highres_patch_size
        self.factor = mlp_factor
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size

        kernel_size=patch_conv_kernel_size
        patch_dim=(highres_patch_size*highres_patch_size)*channels

        self.to_patches=nn.Sequential(
            # view一次
            nn.Conv2d(patch_dim,dim,kernel_size=kernel_size,padding=kernel_size//2)
            # view一次
        )

        #位置嵌入
        num_patches=(image_size//highres_patch_size)**2
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))

        #look-vit
        layers=nn.ModuleList([])
        for _ in range(depth):
            layers.append(nn.ModuleList([
                Attention(dim=dim,heads=heads,dim_head=dim_head,dropout=dropout),
                MLP(dim=dim,hidden_dim=dim*mlp_factor),
                Attention(dim=dim, dim_head=cross_attn_dim_head, heads=cross_attn_heads, dropout=dropout,cross_attend=True),
                Attention(dim=dim, dim_head=cross_attn_dim_head, heads=cross_attn_heads, dropout=dropout,cross_attend=True, reuse_attention=True),
                LayerNorm(dim),
                MLP(dim=dim, hidden_dim=dim * highres_mlp_factor)
            ]))

        self.layers=layers
        self.norm = LayerNorm(dim)
        self.highres_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, num_classes, bias=False)

    def forward(self,x):
        b,c,h,w=x.size()
        p1=p2=self.highres_patch_size
        x=x.view(b,c,h//p1,p1,w//p2,p2).permute(0,3,5,1,2,4).contiguous()
        x=x.view(b,p1*p2*c,h//p1,w//p2).contiguous() #[b,c,h,w]==>[b,p1*p2*c,h//p1,w//p2]

        higher_tokens=self.to_patches(x) #[b,p1*p2*c,h//p1,w//p2]==>[b,dim,h//p1,w//p2]
        higher_tokens=higher_tokens.permute(0,2,3,1)  #[b,dim,h//p1,w//p2]==>[b,h//p1,w//p2,dim]
        higher_tokens=LayerNorm(dim=self.dim)(higher_tokens)

        size=higher_tokens.shape[-2]
        h_, w_ = self.image_size // p1, self.image_size // p2
        pos_emb=position_embed_sin_cos(w=w_,h=h_,dim=self.dim) #[(h//p1)*(w//p2),dim]
        higher_tokens=higher_tokens+pos_emb.view(size,-1,self.dim)  # [b,h//p1,w//p2,dim]

        higher_tokens_2=higher_tokens.permute(0,3,1,2) #[b,h//p1,w//p2,dim]==>[b,dim,h//p1,w//p2]
        #插值
        tokens=nn.functional.interpolate(
            higher_tokens_2,
            w//self.patch_size,  #计算插值后的目标大小  [b,dim,h//p1,w//p2]==>[b,dim,h//p11,w//p22]  p1,p2为高分辨率补丁高（宽）,p11,p22为普通分块的大小
            mode='bilinear'
        )

        b_t,c_t,_,_=tokens.size()
        b_h,_,_,c_h=higher_tokens.size()
        tokens=tokens.permute(0,2,3,1).view(b_t,-1,c_t) #[b,dim,h//p11,w//p22]==>[b,h//p11,w//p22,dim]==>[b,(h*w)//(p11*p22),dim]
        higher_tokens=higher_tokens.view(b_h,-1,c_h) #[b,h//p1,w//p2,dim]==>[b,(h*w)//(p1*p2),dim]

        # look-vit向前传播以及注意力机制
        for attn, mlp, lookup_cross_attn, highres_attn, highres_norm, highres_mlp in self.layers:
            lookup_out,qk_sim=lookup_cross_attn(tokens,higher_tokens,return_qk_sim=True)
            tokens = lookup_out + tokens  # 残差连接：将交叉注意力输出与原始主补丁相加。

            tokens = attn(tokens) + tokens
            tokens = mlp(tokens) + tokens

            qk_sim=qk_sim.transpose(-2,-1) #将注意力矩阵转置，以便进行反向交叉注意力计算。

            higher_tokens=highres_attn(higher_tokens,tokens,qk_sim=qk_sim) + higher_tokens
            higher_tokens=highres_norm(higher_tokens)
            higher_tokens=highres_mlp(higher_tokens) + higher_tokens

        tokens = self.norm(tokens)
        higher_tokens = self.highres_norm(higher_tokens)
        tokens = torch.mean(tokens, dim=1)
        higher_tokens = torch.mean(higher_tokens, dim=1)
        return self.to_logits(tokens + higher_tokens)

test=torch.randn(12,3,32,32)
model=look_vit(dim=64,image_size=32,num_classes=10,patch_size=8)
print(model(test).size())