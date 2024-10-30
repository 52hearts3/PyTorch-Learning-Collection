import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree

class EdgeAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(EdgeAttention, self).__init__()
        self.attention = nn.Linear(in_features, out_features)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, edge_attr):
        attention_scores = self.leakyrelu(self.attention(edge_attr))
        return F.softmax(attention_scores, dim=1)

def start_weight(tensor):
    if tensor is not None:
        torch.nn.init.xavier_uniform_(tensor)
    # Xavier 均匀初始化

def zeros(tensor):
    if tensor is not None:
        torch.nn.init.zeros_(tensor)
class MultiGraphConvolution(MessagePassing): # 多图卷积层
    def __init__(self,ch_in,ch_out,num_edge_features1=3,num_edge_features2=None):
        super(MultiGraphConvolution,self).__init__(aggr='add')

        self.ch_in=ch_in
        self.ch_out=ch_out

        self.weight1 = nn.Parameter(torch.Tensor(ch_in, ch_out))
        self.weight2 = nn.Parameter(torch.Tensor(ch_in, ch_out))
        self.weight3 = nn.Parameter(torch.Tensor(ch_in, ch_out))

        self.edge_attention1 = EdgeAttention(num_edge_features2, ch_out) # 测试不同的边缘特征
        self.edge_attention2 = EdgeAttention(num_edge_features1, ch_out)
        self.edge_attention3 = EdgeAttention(num_edge_features1, ch_out)

        self.attention_mlp=nn.Sequential(
            nn.Linear(ch_out,6),
            nn.ReLU(),
            nn.Linear(6,3),
            nn.ReLU(),
            nn.Linear(3,1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        start_weight(self.weight1)
        start_weight(self.weight2)
        start_weight(self.weight3)
        for layer in self.attention_mlp:
            if isinstance(layer,nn.Linear):
                start_weight(layer.weight)
                zeros(layer.bias)

    def forward(self,x,edge_index1,edge_index2,edge_index3,edge_attr1,edge_attr2,edge_attr3):
        x = F.dropout(x, p=0.5, training=self.training)

        edge_attr1 = self.edge_attention1(edge_attr1)
        edge_attr2 = self.edge_attention2(edge_attr2)
        edge_attr3 = self.edge_attention3(edge_attr3)

        #添加自连接矩阵
        edge_index1,_=add_self_loops(edge_index1,num_nodes=x.size(0)) # 它返回两个值：更新后的边索引和一个包含自环权重的张量
        edge_index2,_=add_self_loops(edge_index2,num_nodes=x.size(0)) # [2,E]==>[2,E+N(节点数)]
        edge_index3,_=add_self_loops(edge_index3,num_nodes=x.size(0))

        # 对新的边缘特征进行扩展
        # 这个张量表示自连接边的特征，每个节点都有一个自连接边，因此有 num_nodes 个自连接边，每个边的特征维度为 self.ch_out。
        num_nodes = x.size(0)
        self_loop_attr = torch.zeros((num_nodes, self.ch_out), device=x.device)
        edge_attr1 = torch.cat([edge_attr1, self_loop_attr], dim=0)
        edge_attr2 = torch.cat([edge_attr2, self_loop_attr], dim=0)
        edge_attr3 = torch.cat([edge_attr3, self_loop_attr], dim=0)

        row1,col1=edge_index1  #取第一行，第二行
        row2,col2=edge_index2
        row3,col3=edge_index3

        deg1 = degree(row1, x.size(0), dtype=x.dtype) # 求度矩阵
        deg2 = degree(row2, x.size(0), dtype=x.dtype) # 度矩阵用于表示每个节点的连接数
        deg3 = degree(row3, x.size(0), dtype=x.dtype) # [num_features]

        norm1 = deg1.pow(-0.5)
        norm2 = deg2.pow(-0.5)
        norm3 = deg3.pow(-0.5)

        norm1 = norm1[row1] * norm1[col1] # 根据边索引进行归一化：
        norm2 = norm2[row2] * norm2[col2]
        norm3 = norm3[row3] * norm3[col3]

        x1 = torch.matmul(x, self.weight1)  # [x,num_features(ch_in)]*[ch_in,ch_out]
        x2 = torch.matmul(x, self.weight2)
        x3 = torch.matmul(x, self.weight3)


        out1 = self.propagate(edge_index1, x=x1, norm=norm1,edge_attr=edge_attr1)  # [x,ch_out]
        out2 = self.propagate(edge_index2, x=x2, norm=norm2,edge_attr=edge_attr2)
        out3 = self.propagate(edge_index3, x=x3, norm=norm3,edge_attr=edge_attr3)

        gap1 = self.graph_gap(out1, edge_index1)
        gap2 = self.graph_gap(out2, edge_index2)
        gap3 = self.graph_gap(out3, edge_index3)

        att1 = self.attention_mlp(gap1) # [x,1]
        att2 = self.attention_mlp(gap2)
        att3 = self.attention_mlp(gap3)

        alpha = F.softmax(torch.cat([att1, att2, att3]), dim=0) #[3x,1]
        out = alpha[0] * out1 + alpha[1] * out2 + alpha[2] * out3  # [x,1]==>(广播)[x,x]*[x,ch_out]
        return out
    def message(self, x_j,norm,edge_attr):
        # x_j  [num_edges, ch_out]
        # print(edge_attr.size())
        # print((norm.view(-1,1)*x_j).size())
        return norm.view(-1,1)*x_j * edge_attr

    def update(self, aggr_out):
        return aggr_out

    # 计算图神经网络中节点特征的全局平均池化
    def graph_gap(self, x, edge_index):
        row,col=edge_index

        deg=degree(row,x.size(0),dtype=x.dtype) #形如[num_nodes]
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        # 计算了每个节点的度，并求其倒数。如果度为零，则将倒数设为零以避免无穷大。

        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, row, x[col])
        # 如 x形如 [1,2]   edge_index形如 [0,1,2]    neighbor_sum形如 [0,0]
        #         [3,4]                 [1,2,0]                    [0,0]
        #         [5,6]                                            [0,0]
        # 这里 row = [0, 1, 2] 和 col = [1, 2, 0]
        # x[col] 会得到 [3, 4]   然后 index_add_ 会将这些值累加到 neighbor_sum 的对应行中
        #              [5, 6]
        #              [1, 2]
        x_avg = deg_inv.view(-1, 1) * neighbor_sum  # deg_inv [num_nodes,1]被广播为[num_nodes,num_features]
        return torch.mean(x_avg, dim=0)



# 生成简单数据以测试模型
num_nodes = 4
num_node_features = 5
num_edge_features = 3
num_edge_features2 = 5
# 节点特征（随机生成用于测试）
x = torch.randn((num_nodes, num_node_features))
# 边索引（随机生成用于测试）
edge_index1 = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 3],[3,1]], dtype=torch.long).t().contiguous()
edge_index2 = torch.tensor([[0, 2], [2, 3], [3, 0], [0, 1]], dtype=torch.long).t().contiguous()
edge_index3 = torch.tensor([[0, 3], [3, 1], [1, 0], [0, 2]], dtype=torch.long).t().contiguous()

# 边特征（随机生成用于测试）
edge_attr1 = torch.randn((edge_index1.size(1), num_edge_features2)) #[4,3]
edge_attr2 = torch.randn((edge_index2.size(1), num_edge_features))
edge_attr3 = torch.randn((edge_index3.size(1), num_edge_features))

# 初始化模型
model = MultiGraphConvolution(ch_in=num_node_features, ch_out=8,num_edge_features2=5)

# 通过模型进行前向传播
output = model(x=x,
               edge_index1=edge_index1,
               edge_attr1=edge_attr1,
               edge_index2=edge_index2,
               edge_attr2=edge_attr2,
               edge_index3=edge_index3,
               edge_attr3=edge_attr3)

print("模型输出：")
print(output.size())

#基于相似性：
#计算节点特征（如基因表达）的相似性，使用相似性度量（如皮尔逊相关系数、余弦相似度等）来确定连接。
#例如，如果两个节点的基因表达模式非常相似，可以在它们之间添加一条边。
# import torch
# import numpy as np
#
# # 假设 gene_expression 是形如 [num_nodes, num_genes] 的张量
# gene_expression = torch.tensor([...], dtype=torch.float)
#
# # 计算相似性矩阵（例如，使用余弦相似度）
# similarity_matrix = torch.mm(gene_expression, gene_expression.t())
#
# # 设置阈值，确定哪些节点之间有边
# threshold = 0.8
# edge_index = (similarity_matrix > threshold).nonzero(as_tuple=False).t()


#基于距离：
#使用节点特征的距离（如欧氏距离）来确定连接。
#例如，如果两个节点的特征距离在某个阈值内，可以在它们之间添加一条边。

# from scipy.spatial.distance import pdist, squareform
#
# # 计算距离矩阵（例如，使用欧氏距离）
# distance_matrix = squareform(pdist(gene_expression.numpy(), metric='euclidean'))
#
# # 设置阈值，确定哪些节点之间有边
# threshold = 1.0
# edge_index = (distance_matrix < threshold).nonzero()
# edge_index = torch.tensor(edge_index, dtype=torch.long).t()


# 组织特异基因
# 1 训练模型以预测或分类不同组织区域。
# 2 分析模型的注意力权重和节点特征，识别在特定区域内高表达的基因。

# 空间暗基因
# 1 训练模型以捕捉空间位置上的基因表达变化。
# 2 分析模型的输出和注意力机制，识别在特定空间位置上显著变化的基因。

# 提取注意力权重和节点特征
# model.eval()
# with torch.no_grad():
#     for batch in loader:
#         out = model(batch.x, batch.edge_index1, batch.edge_index2, batch.edge_index3,
#                     batch.edge_attr1, batch.edge_attr2, batch.edge_attr3)
#         attention_weights1 = model.edge_attention1.leakyrelu(model.edge_attention1.attention(batch.edge_attr1))
#         attention_weights2 = model.edge_attention2.leakyrelu(model.edge_attention2.attention(batch.edge_attr2))
#         attention_weights3 = model.edge_attention3.leakyrelu(model.edge_attention3.attention(batch.edge_attr3))
#         updated_node_features = out
#
# # 可视化注意力权重和节点特征
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Attention Weights')
# plt.plot(attention_weights1.cpu().numpy(), label='Edge 1')
# plt.plot(attention_weights2.cpu().numpy(), label='Edge 2')
# plt.plot(attention_weights3.cpu().numpy(), label='Edge 3')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.title('Updated Node Features')
# plt.plot(updated_node_features.cpu().numpy())
# plt.show()