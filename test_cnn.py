import torch
import torch.nn as nn
from struct import unpack
import gzip
import numpy as np

# 定义你训练时候的小CNN（一定要和train的一样）
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = SmallCNN().to(device)

# 加载.pth
model.load_state_dict(torch.load('./saved_models/best_cnn.pth', map_location=device))

model.eval()

# 加载测试集
test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 归一化 + 转成Tensor
test_imgs = torch.tensor(test_imgs / 255.0, dtype=torch.float32).to(device)
test_labs = torch.tensor(test_labs, dtype=torch.long).to(device)

# 前向推理
with torch.no_grad():
    outputs = model(test_imgs)
    preds = outputs.argmax(dim=1)
    correct = (preds == test_labs).sum().item()
    acc = correct / test_labs.size(0)

print(f'Test Accuracy: {acc:.4f}')