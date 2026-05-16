import torch#导入torch库
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn import CNNNet
import torch.nn as nn#导入nn功能
# 为什么导入了torch还要再导入里面的小库呢？
# 因为torch里包扩torch.utils.data，但是不不包括torch.utils.data里的dataloader
# 数据预处理
transform = transforms.ToTensor()#这是用在告诉下面的dataloader要把图片转成张量

# 下载数据
train_dataset = datasets.MNIST(#选择训练数据集
    root="./data",#根目录下面也一样
    train=True,#用于训练
    transform=transform,#转张量
    download=True#不是本地数据集，下载
)

test_dataset = datasets.MNIST(#选择测试数据集
    root="./data",#根目录
    train=False,#不用于训练
    transform=transform,#转张量
    download=True#不是本地数据集
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)#导入训练数据集，batchsize（一次导入）1024张图，shuffle（打乱）✅，子进程4个
test_loader = DataLoader(test_dataset, batch_size=1024)#导入测试数据集，1024张图一次

#%%
model = CNNNet()#定义model为一个CNNNet对象（你上面定义的）

loss_fn = nn.CrossEntropyLoss()#创建一个交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#创建一个 Adam 优化器，用来更新模型参数，

#%%
device = torch.device("mps")#定义device为的苹果gpu加速后段
model.to(device)#将网络移植到设备上
losses = []#创建损失函数的结果列表
for epoch in range(20):#20轮
    print("epoch:", epoch)
    for images, labels in train_loader:#按batch读取数据
        images = images.to(device)  #图片移动到gpu
        labels = labels.to(device)  #标签移动到gpu
        outputs = model(images)#前向传播
        loss = loss_fn(outputs, labels)#计算loss
        print("epoch",epoch,"loss",loss.item())
        losses.append(loss.item())#记录loss
        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播
        optimizer.step()#更新参数

torch.save(model.state_dict(), "mnist_cnn.pth")