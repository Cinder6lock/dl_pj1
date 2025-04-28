# train_cnn_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import gzip
from struct import unpack
import numpy as np
import pickle
import os

# 固定随机种子
torch.manual_seed(309)
np.random.seed(309)

# 读取MNIST数据
train_images_path = './dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = './dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 打乱数据并保存idx
idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

# 归一化到[0,1]
train_imgs = train_imgs / 255.0

# 转成Tensor
train_imgs = torch.tensor(train_imgs, dtype=torch.float32).unsqueeze(1)  # (batch, 1, 28, 28)
train_labs = torch.tensor(train_labs, dtype=torch.long)

# 划分训练集和验证集
train_dataset = TensorDataset(train_imgs[10000:], train_labs[10000:])
valid_dataset = TensorDataset(train_imgs[:10000], train_labs[:10000])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

# 定义CNN模型
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 28x28 → 28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 → 14x14

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 14x14 → 14x14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 → 7x7

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('mps' if torch.mps.is_available() else 'cpu')
model = SmallCNN().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 训练函数
def train(model, optimizer, criterion, scheduler, train_loader, valid_loader, num_epochs=20, patience=5):
    best_val_acc = 0
    trigger_times = 0

    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for batch_imgs, batch_labels in train_loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # 验证
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for batch_imgs, batch_labels in valid_loader:
                batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        avg_val_loss = val_loss / len(valid_loader)
        val_acc = correct / total
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}  Val Acc: {val_acc:.4f}')

        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
            torch.save(model.state_dict(), './saved_models/best_cnn.pth')
            import pickle
            with open('./saved_models/best_cnn.pkl', 'wb') as f:
                pickle.dump(model.state_dict(), f)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    return train_loss_history, val_loss_history, val_acc_history

# 开始训练
train_loss, val_loss, val_acc = train(model, optimizer, criterion, scheduler, train_loader, valid_loader)

# 画图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(val_acc, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy Curve')

plt.tight_layout()
plt.show()