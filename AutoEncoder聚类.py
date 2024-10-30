import torch
from torch import nn,optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Attention_block(nn.Module):
    def __init__(self,ch_in):
        super(Attention_block,self).__init__()

        self.ch_in=ch_in
        self.to_qkv=nn.Conv2d(ch_in,ch_in*3,kernel_size=1,stride=1,padding=0)
        self.to_out=nn.Conv2d(ch_in,ch_in,kernel_size=1)

        self.norm=nn.Sequential(
            nn.BatchNorm2d(ch_in)
        )

    def forward(self,x):
        b,ch,h,w=x.size()
        x_norm=self.norm(x)
        x_qkv=self.to_qkv(x_norm)
        q,k,v=torch.split(x_qkv,self.ch_in,dim=1)
        q=q.permute(0,2,3,1).view(b,h*w,ch)
        k=k.view(b,ch,h*w)
        v=v.permute(0,2,3,1).view(b,h*w,ch)

        dot=torch.bmm(q,k)*(ch**-0.5)
        attention=torch.softmax(dot,dim=-1)
        out=torch.bmm(attention,v).view(b,h,w,ch).permute(0,3,1,2)
        return self.to_out(out)+x

test=torch.randn(12,3,32,32)
model=Attention_block(ch_in=3)
print(model(test).size())

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk,self).__init__()

        self.ch_in=ch_in

        self.conv=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

        self.attention=Attention_block(ch_out)
        self.shortcut=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self,x):
        out=self.conv(x)
        out=self.attention(out)
        x=out+self.shortcut(x)
        return x

test=torch.randn(12,3,32,32)
model=ResBlk(ch_in=3,ch_out=16)
print(model(test).size())
class up_sample(nn.Module):
    def __init__(self,ch_in):
        super(up_sample,self).__init__()

        self.up_sample=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=1,padding=1)
        )

    def forward(self,x):
        x=self.up_sample(x)
        return x

class down_sample(nn.Module):
    def __init__(self,ch_in):
        super(down_sample,self).__init__()

        self.down_sample=nn.Sequential(
            nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.down_sample(x)
        return x

class Unet(nn.Module):
    def __init__(self,ch_in,dim):
        super(Unet,self).__init__()

        self.conv1=nn.Conv2d(ch_in,dim,kernel_size=3,stride=1,padding=1)

        #下采样 每一次加入两个短接层
        down=[]
        for i in range(4):
            for _ in range(2):
                down.append(ResBlk(ch_in=dim,ch_out=dim))
            down.append(down_sample(ch_in=dim))

        self.down_sample=nn.Sequential(*down)

        self.mid_layer=nn.Sequential(
            ResBlk(ch_in=dim,ch_out=dim),
            ResBlk(ch_in=dim,ch_out=dim)
        )

        #上采样 每一次加入两个短接层
        up=[]
        for i in range(4):
            for _ in range(2):
                up.append(ResBlk(ch_in=dim,ch_out=dim))
            up.append(up_sample(ch_in=dim))

        self.up_sample=nn.Sequential(*up)

        self.change_ch = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, ch_in, 3, padding=1, stride=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.conv1(x)
        #下采样
        down_output=[]
        for layer in self.down_sample:
            x=layer(x)
            down_output.append(x)

        x=self.mid_layer(x)
        #上采样
        for layer in self.up_sample:
            skip_connection = down_output.pop()
            if skip_connection.size() != x.size():
                skip_connection = torch.nn.functional.interpolate(skip_connection, size=x.size()[2:])
            x = torch.cat((x, skip_connection), dim=1)  # [b,ch,x,x]==>[b,2ch,x,x]
            x = self.change_ch(x)
            # print('s',x.size())
            # print(x.size())
            x = layer(x)

        x=self.out_conv(x)
        return x

test=torch.randn(12,3,32,32)
model=Unet(ch_in=3,dim=64)
print(model(test).size())

class AutoEncoder(nn.Module):
    def __init__(self,ch_in,dim):
        super(AutoEncoder,self).__init__()
        self.unet=Unet(ch_in=ch_in,dim=dim)
        self.encoder=nn.Sequential(
            nn.Linear(3*32*32,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,20)
        )
        self.decoder=nn.Sequential(
            nn.Linear(20,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,3*32*32)
        )

    def forward(self,x):
        x_unet=self.unet(x)
        #print(x.size())
        x = x_unet.contiguous().view(x.size(0), -1)  # 使用 contiguous() 确保张量连续
        z = self.encoder(x)
        x_recon = self.decoder(z)
        x_recon=x_recon.view(x.size(0),3,32,32)
        return x_recon, z




# 预训练自编码器
def pretrain_autoencoder(autoencoder, data_loader, epochs=1, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in data_loader:
            images, _ = batch
            images=images.to(device)
            optimizer.zero_grad()
            x_recon, _ = autoencoder(images)
            loss = criterion(x_recon, images)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# 特征提取
def extract_features(autoencoder, data_loader):
    features = []
    with torch.no_grad():
        for batch in data_loader:
            images, _ = batch
            images=images.to(device)
            _,z = autoencoder(images)
            features.append(z.view(z.size(0), -1))
    return torch.cat(features)

# 使用肘部法则确定最佳聚类数量
def find_optimal_clusters(features, max_k):
    iters = range(2, max_k+1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)
    plt.plot(iters, sse, marker='o')
    plt.xlabel('Cluster Centers')
    plt.ylabel('SSE')
    plt.show()

# 使用轮廓系数确定最佳聚类数量
def find_optimal_clusters_silhouette(features, max_k):
    iters = range(2, max_k+1)
    silhouette_scores = []
    for k in iters:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)
    plt.plot(iters, silhouette_scores, marker='o')
    plt.xlabel('Cluster Centers')
    plt.ylabel('Silhouette Score')
    plt.show()


# 使用DBSCAN进行聚类
def dbscan_clustering(features, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_scaled)

    # 可视化聚类结果
    plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('DBSCAN Clustering')
    plt.show()
    return labels
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载MNIST数据集
data = datasets.CIFAR10(root=r'D:\game\pytorch\简单分类问题\data', train=True, transform=transform, download=True)
subset_indices = list(range(1000))
subset = Subset(data, subset_indices) #只取用2000个训练，防止聚类时卡死
data_loader = DataLoader(subset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder =AutoEncoder(ch_in=3,dim=64).to(device)
pretrain_autoencoder(autoencoder, data_loader, epochs=10)

# 提取特征
features = extract_features(autoencoder, data_loader).cpu().numpy()
print(features.shape)
# 确定最佳聚类数量
# find_optimal_clusters(features, max_k=20)
# find_optimal_clusters_silhouette(features, max_k=20)

# 使用DBSCAN进行聚类
labels = dbscan_clustering(features, eps=0.5, min_samples=5)