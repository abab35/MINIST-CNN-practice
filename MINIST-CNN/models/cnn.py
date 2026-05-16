#%%
import torch.nn as nn#导入nn功能


class CNNNet(nn.Module):#创建一个CNN-NET符合nn.Module的pytorch规则
    def __init__(self):#构造函数__init__,参数self
        super().__init__()#先把nn中的__init__函数搬过来，然后下面的东西你自己写

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 图片卷积，图片尺寸(1,28,28) → (16,28,28)
        self.relu = nn.ReLU()#定义激活函数
        self.pool = nn.MaxPool2d(2)#最大池化，边长尺寸为2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  #(16,28,28)->(32,28,28)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=11, padding=4)  #(32,28,28->64,28,28)
        self.fc = nn.Linear(64 * 13 * 13, 10)#线性层

    def forward(self, x):#定义了前向传播函数
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x