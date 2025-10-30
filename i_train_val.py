import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 定义简化的 YOLO 网络
class SimplifiedYOLO(nn.Module):
    def __init__(self):
        super(SimplifiedYOLO, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 6 * 5, 20)  # 根据卷积和池化后的尺寸调整

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 5)
        x = self.fc(x)
        return x

# 定义全连接网络处理后 40 个数据
class FullyConnectedLayer(nn.Module):
    def __init__(self):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(40, 20)

    def forward(self, x):
        return self.fc(x)

# 定义完整的神经网络模型
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.yolo_part = SimplifiedYOLO()
        self.fc_part = FullyConnectedLayer()
        # 根据 label 的实际类别数调整输出维度，这里假设类别数为 10，根据实际情况修改
        self.classifier = nn.Linear(40, 10)  

    def forward(self, x):
        data_500 = x[:, :500].view(-1, 1, 25, 20).float()  # 转换为 2D 图像格式
        data_40 = x[:, 500:].float()

        yolo_output = self.yolo_part(data_500)
        fc_output = self.fc_part(data_40)

        combined = torch.cat((yolo_output, fc_output), dim=1)
        output = self.classifier(combined)
        return output

# 读取 CSV 文件
df = pd.read_csv('./data_with_feature/data_add_feature.csv')

# 提取 data 和 label
data = df['data'].apply(lambda x: eval(x)).to_list()
data = torch.tensor(data)
label = torch.tensor(df['label'].values)

# 打乱数据集
indices = np.arange(len(data))
np.random.shuffle(indices)
data = data[indices]
label = label[indices]

# 划分训练集和测试集（简单示例，按 80:20 划分）
train_size = int(0.8 * len(data))
train_data, test_data = data[:train_size], data[train_size:]
train_label, test_label = label[:train_size], label[train_size:]

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器
model = CombinedModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 90
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_data, batch_label in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_data, batch_label in test_loader:
        outputs = model(batch_data)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_label.size(0)
        correct += (predicted == batch_label).sum().item()

print(f'Accuracy on the test set: {100 * correct / total}%')