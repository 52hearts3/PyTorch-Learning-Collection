import torch
from torch import nn,optim

def position_embed_sin_cos_2d(w,h,dim,temperature=1000):
    y,x=torch.meshgrid(torch.arange(w),torch.arange(h),indexing='ij')
    omega=torch.arange(dim//4)/(dim//4-1)
    omega=1.0/(temperature**omega)
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
    def __init__(self,dim,dim_hidden):
        super(MLP,self).__init__()

        self.net=nn.Sequential(
            LayerNorm(dim=dim),
            nn.Linear(dim,dim_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim_hidden,dim),
            nn.Dropout(0.2)
        )

    def forward(self,x):
        x=self.net(x)
        return x

class Attention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.,cross_attend=False,reuse_attention=False):
        super(Attention,self).__init__()

        inner_dim=dim_head*dim
        self.scale=dim_head**-0.5

        self.heads = heads
        self.reuse_attention = reuse_attention
        self.cross_attend = cross_attend

        self.norm=LayerNorm(dim) if reuse_attention==True else nn.Identity()
        self.norm_context=LayerNorm(dim) if cross_attend==True else nn.Identity()

        self.attend=nn.Softmax(-1)
        self.drop_out=nn.Dropout(dropout)

        self.to_q=nn.Linear(dim,inner_dim,bias=False) if reuse_attention==True else None
        self.to_k=nn.Linear(dim,inner_dim,bias=False) if reuse_attention==True else None
        self.to_v=nn.Linear(dim,inner_dim,bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
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
        if self.reuse_attention:
            q,k=self.to_q(x),self.to_k(x)
            q = q.view(b, n, self.heads, -1).permute(0, 2, 1, 3).contiguous()
            k = k.view(b, n, self.heads, -1).permute(0, 2, 1, 3).contiguous()
            q=q*self.scale
            qk_sim=torch.matmul(q,k.transpose(-2,-1))
            #[b,heads,n,hd/heads]*[b,heads,hd/heads,n]==>[b,heads,n,n]

        attend=self.attend(qk_sim)
        attention=self.drop_out(attend)
        dot=torch.matmul(attention,v) #==>[b,heads,n,hd/heads]
        out = dot.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        out=self.to_out(out)
        if return_qk_sim == True:
            return out,qk_sim
        return out

test=torch.randn(12,3,8*8)
model=Attention(dim=64,cross_attend=True,reuse_attention=True)
context=torch.randn_like(test)
print(model(test,context).size())

class Transformer(nn.Module):
    def __init__(self,dim,hidden_dim,depth=5):
        super(Transformer,self).__init__()

        self.norm=LayerNorm(dim)

        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim,heads=8,reuse_attention=True),
                MLP(dim=dim,dim_hidden=hidden_dim),
                Attention(dim=dim, heads=8,reuse_attention=True),
                Attention(dim=dim, heads=8,reuse_attention=True),
                LayerNorm(dim),
                MLP(dim=dim,dim_hidden=hidden_dim)
            ]))

    def forward(self,x):
        for model in self.layers:
            for layer in model:
                if isinstance(layer,Attention):
                    x=layer(x)+x
                elif isinstance(layer,MLP):
                    x=layer(x)+x
                else:
                    x=layer(x)
        return self.norm(x)

test=torch.randn(12,3,8*8)
model=Transformer(dim=64,hidden_dim=64*4)
print(model(test).size())

class Simple_look_vit(nn.Module):
    def __init__(self,image_h,image_w,dim,patch_h,patch_w,num_classes,ch_in=3):
        super(Simple_look_vit,self).__init__()

        self.patch_h = patch_h
        self.patch_w = patch_w

        patch_dim=ch_in*patch_h*patch_w
        self.to_patch_embedding=nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim,dim),
            LayerNorm(dim)
        )

        self.pos_embedding=position_embed_sin_cos_2d(
            w=image_w/patch_w,
            h=image_h/patch_h,
            dim=dim
        )

        self.transformer = Transformer(
            dim=dim,
            hidden_dim=64,
        )

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self,x):
        device=x.device
        b,c,h,w=x.size()
        p1,p2=self.patch_h,self.patch_w
        x=x.view(b,c,h//p1,p1,w//p2,p2).permute(0, 2, 4, 3, 5, 1).contiguous()
        x=x.view(b,-1,p1*p2*c)
        x=self.to_patch_embedding(x)
        x = x + self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.linear_head(x)

test=torch.randn(12,3,32,32)
model=Simple_look_vit(image_h=32,image_w=32,dim=64,patch_w=8,patch_h=8,num_classes=10)
print(model(test).size())