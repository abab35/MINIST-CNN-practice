from models.cnn import CNNNet
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
model = CNNNet()
model.load_state_dict(torch.load("mnist_cnn.pth"))
correct = 0  # 正确数量

total = 0  # 总数量

model.eval()  # 切换测试模式
device = torch.device("mps")#定义device为的苹果gpu加速后段
model.to(device)#将网络移植到设备上
with torch.no_grad():  # 关闭梯度计算

    for images, labels in test_loader:  # 读取测试数据

        images = images.to(device)  # 图片移动到GPU
        labels = labels.to(device)  # 标签移动到GPU

        outputs = model(images)  # 前向传播

        _, predicted = torch.max(outputs, 1)  # 获取预测类别

        total += labels.size(0)  # 累加总数

        correct += (predicted == labels).sum().item()  # 累加正确数

print("accuracy:", correct / total)  # 输出准确率2