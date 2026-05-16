import torch
from torchvision import datasets, transforms
from models.cnn import CNNNet

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 数据预处理
transform = transforms.ToTensor()

# 测试数据集
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# 取一张图片
import random

index = random.randint(0, len(test_dataset)-1)

image, label = test_dataset[index]

# 增加 batch 维度
image = image.unsqueeze(0)

# 移动到 device
image = image.to(device)

# 加载模型
model = CNNNet().to(device)

model.load_state_dict(torch.load("mnist_cnn.pth"))

# 切换到测试模式
model.eval()

# 关闭梯度
with torch.no_grad():

    outputs = model(image)

    _, predicted = torch.max(outputs, 1)

print("Prediction:", predicted.item())

print("True Label:", label)

import matplotlib.pyplot as plt

plt.imshow(image.cpu().squeeze(), cmap="gray")

plt.title(f"Prediction: {predicted.item()}")

plt.show()