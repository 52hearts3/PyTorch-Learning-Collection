import torch
from torch.nn import functional
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
batch_sizes=200
train_loader = DataLoader(datasets.MNIST(root='./data', train=True, transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307),(0.3081))]), download=True)
                                    ,batch_size=batch_sizes,shuffle=True)
test_loader = DataLoader(datasets.MNIST(root='./data', train=False,
                                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307),(0.3081))]))
                                   ,batch_size=batch_sizes,shuffle=True)

w1,b1=torch.randn(200,784,requires_grad=True),torch.zeros(200,requires_grad=True)
#线性层1
w2,b2=torch.randn(200,200,requires_grad=True),torch.zeros(200,requires_grad=True)
#线性层2
w3,b3=torch.randn(10,200,requires_grad=True),torch.zeros(10,requires_grad=True)
#线性层3  最后是10分类，所以输出节点为10

#初始化
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)
def forward(x):
    x=x@w1.t()+b1
    x=functional.relu(x)
    x=x@w2.t()+b2
    x=functional.relu(x)
    x=x@w3.t()+b3
    x=functional.relu(x)
    return x
#定义优化器
optimizer=torch.optim.SGD([w1,b1,w2,b2,w3,b3],lr=0.01,momentum=0.78)
crition=nn.CrossEntropyLoss()


epochs=10
for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
      data=data.view(-1,28*28)
      logits=forward(data)
      loss=crition(logits,target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx%100==0:
          print('train epoch:{}[{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(epoch,batch_idx*len(data),len(train_loader.dataset)
                                                                       ,100.*batch_idx/len(train_loader),loss.item()))
    test_loss=0
    correct=0
    for data,target in test_loader:
        data=data.view(-1,28*28)
        logits=forward(data)
        test_loss+=crition(logits,target).item()
        pred=logits.data.max(1)[1]
        correct+=pred.eq(target.data).sum()
    test_loss/=len(test_loader.dataset)
    print('\ntest set : average loss: {:.4f}, accuracy : {}/{} ({:.0f}%)\n'.format(test_loss,correct,len(test_loader.dataset)
                                                                                   ,100.*correct/len(test_loader.dataset)))
