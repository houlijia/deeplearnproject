import torch
import torch.nn as nn
import torch.optim as optim


class SimlpleNN(nn.Module):
    """
    在 PyTorch 中，构建神经网络通常需要继承 nn.Module 类。
    nn.Module 是所有神经网络模块的基类，你需要定义以下两个部分：
    __init__()：定义网络层。
    forward()：定义数据的前向传播过程。
    """

    def __init__(self):
        super(SimlpleNN, self).__init__()
        # define fully connected layers
        self.fc1 = nn.Linear(2, 2)  # input size: 2, output size: 2
        self.fc2 = nn.Linear(2, 1)  # input size: 2, output size: 1

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimlpleNN()
# print(model)

# 训练模型
print('Training model......')
# 优化器（Optimizer）
# 1、使用 SGD 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 2、使用 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 损失函数（Loss Function）
# 1、均方误差损失
criterion = nn.MSELoss()
# 2、交叉熵损失
criterion = nn.CrossEntropyLoss()
# 3、二分类交叉熵损失
criterion = nn.BCEWithLogitsLoss()

# 训练数据示例
X = torch.randn(10, 2)  # 10 个样本，每个样本有 2 个特征
Y = torch.randn(10, 1)  # 10 个目标标签
print(f'Training model with input X: {X}')
print(f'Training model with input Y: {Y}')

# 训练过程
for epoch in range(100):  # 训练 100 轮
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清除梯度
    output = model(X)  # 前向传播
    loss = criterion(output, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    if (epoch + 1) % 10 == 0:  # 每 10 轮输出一次损失
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
