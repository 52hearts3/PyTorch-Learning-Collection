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

# test=torch.randn(12,3,8*8)
# model=Attention(dim=64,cross_attend=True,reuse_attention=True)
# context=torch.randn_like(test)
# print(model(test,context).size())

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

# test=torch.randn(12,3,8*8)
# model=Transformer(dim=64,hidden_dim=64*4)
# print(model(test).size())

class DetectionHead(nn.Module):
    def __init__(self, dim, num_classes, num_boxes):
        super(DetectionHead, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.fc = nn.Linear(dim, num_boxes * (num_classes + 4))  # 4 for bbox coordinates

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.num_boxes, self.num_classes + 4)
        return x

class Simple_look_vit(nn.Module):
    def __init__(self,image_h,image_w,dim,patch_h,patch_w,num_classes,num_boxes,ch_in=3):
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

        self.detection_head = DetectionHead(dim, num_classes, num_boxes)

    def forward(self,x):
        device=x.device
        b,c,h,w=x.size()
        p1,p2=self.patch_h,self.patch_w
        x=x.view(b,c,h//p1,p1,w//p2,p2).permute(0, 2, 4, 3, 5, 1).contiguous()
        x=x.view(b,-1,p1*p2*c)
        x=self.to_patch_embedding(x)
        # print('x',x.size())
        # print('pos embed',self.pos_embedding.to(device, dtype=x.dtype).size())
        x = x + self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.detection_head(x)

# test=torch.randn(12,3,32,32)
# model=Simple_look_vit(image_h=32,image_w=32,dim=64,patch_w=8,patch_h=8,num_classes=10,num_boxes=3)
# print(model(test).size())

import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection

# 自定义collate函数
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets


# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),  # 数据增强
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 数据增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载VOC 2007数据集
dataset = VOCDetection(root=r'D:\game\pytorch\ViT-Transformer\VOC2007', year='2007', image_set='train', download=True, transform=transform)
val_dataset = VOCDetection(root=r'D:\game\pytorch\ViT-Transformer\VOC2007', year='2007', image_set='val', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=collate_fn)
# 定义类别名称到整数标签的映射
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# 使用VOC2007训练集的类别样本计数
class_sample_counts = [2501, 1411, 1447, 788, 1172, 1097, 2850, 1371, 1532, 356, 1090, 1382, 1394, 1410, 4690, 1250, 1095, 780, 1466, 1365]
total_samples = sum(class_sample_counts)
class_weights = [total_samples / count for count in class_sample_counts]
class_weights = torch.tensor(class_weights, device='cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型
model = Simple_look_vit(image_h=64, image_w=64, dim=64, patch_h=8, patch_w=8, num_classes=20, num_boxes=10)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 定义损失函数和优化器
classification_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)  # 使用加权二元交叉熵损失
bbox_criterion = nn.SmoothL1Loss()  # 使用平滑L1损失计算边界框回归损失
optimizer = optim.Adam(model.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调度器

def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(images)
            class_outputs = outputs[:, :, :20].sigmoid()

            for i, target in enumerate(targets):
                objs = target['annotation']['object']
                if not isinstance(objs, list):
                    objs = [objs]
                for obj in objs:
                    class_idx = class_to_idx[obj['name']]
                    if class_outputs[i, :, class_idx].max() > 0.5:  # 使用阈值0.5判断是否预测为该类别
                        correct += 1
                    total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy



def train():
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, targets) in enumerate(dataloader):
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')

            # 提取目标信息并转换为张量
            labels = torch.zeros((images.size(0), 10, 20), dtype=torch.float).to(
                'cuda' if torch.cuda.is_available() else 'cpu')
            bboxes = torch.zeros((images.size(0), 10, 4), dtype=torch.float).to(
                'cuda' if torch.cuda.is_available() else 'cpu')
            for j, target in enumerate(targets):
                objs = target['annotation']['object']
                if not isinstance(objs, list):
                    objs = [objs]
                for k, obj in enumerate(objs):
                    if k >= 10:  # 确保不会超过预设的边界框数量
                        break
                    class_idx = class_to_idx[obj['name']]
                    assert 0 <= class_idx < 20, f"Class index out of range: {class_idx}"
                    labels[j, k, class_idx] = 1
                    bbox = obj['bndbox']
                    bboxes[j, k] = torch.tensor(
                        [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])

            # 前向传播
            outputs = model(images)
            class_outputs = outputs[:, :, :20]  # 假设20个类别
            bbox_outputs = outputs[:, :, 20:]  # 边界框输出

            # 计算损失
            class_loss = classification_criterion(class_outputs, labels)
            bbox_loss = bbox_criterion(bbox_outputs.view(-1, 4), bboxes.view(-1, 4))
            loss = class_loss +  bbox_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 9:  # 每100个批次打印一次
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # 打印预测类别分布
        predicted_class_counts = torch.bincount(torch.argmax(class_outputs.sigmoid(), dim=-1).view(-1), minlength=20)
        print(f"Batch [{i + 1}/{len(dataloader)}] Predicted Class Distribution: {predicted_class_counts}")

        # 验证模型并计算准确率
        accuracy = validate()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')

        scheduler.step()  # 更新学习率
        if epoch % 10 == 0:
            test_and_visualize()
    print('Finished Training')


import matplotlib.pyplot as  plt
import numpy as np
# 可视化结果
def visualize_prediction(image, output):
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # 反归一化
    image = np.clip(image, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # 获取预测的类别和边界框
    class_outputs = output[:, :20].sigmoid()
    bbox_outputs = output[:, 20:]

    # 取最大值所在的索引作为标签
    predicted_labels = torch.argmax(class_outputs, dim=-1)
    bboxes = bbox_outputs.cpu().numpy()

    # 设置一个阈值，只显示概率大于阈值的预测
    threshold = 0.5
    for i in range(predicted_labels.size(0)):
        class_idx = predicted_labels[i].item()
        if class_outputs[i, class_idx] > threshold:
            xmin, ymin, xmax, ymax = bboxes[i]
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red')
            ax.add_patch(rect)
            ax.text(xmin, ymin, class_names[class_idx], bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()

def test_and_visualize():
    # 加载VOC 2007测试数据集
    test_dataset = VOCDetection(root=r'D:\game\pytorch\ViT-Transformer\VOC2007', year='2007', image_set='val',
                                download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    # 获取一个测试样本
    images, targets = next(iter(test_dataloader))
    images = images.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(images)

    # 可视化第一个测试样本的预测结果
    visualize_prediction(images[0], outputs[0])


if __name__ == '__main__':
    train()