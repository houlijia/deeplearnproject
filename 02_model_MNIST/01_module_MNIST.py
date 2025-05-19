import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义超参数
BATCH_SIZE = 10  # 每批处理的数据
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # 启用mps
EPOCHS = 10  # 训练数据的轮次

# 构建pipeline，对图形进行处理
pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 正则化：降低模型复杂度
])

# 下载数据
train_set = datasets.MNIST(root='data', train=True, download=True, transform=pipeline)
test_set = datasets.MNIST(root='data', train=False, download=True, transform=pipeline)

# 加载数据集
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


# 构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # 1: 灰度图像通道, 10:输出通道, 5:kernel
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10: 输入通道, 20 输出通道, 3:kernel
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 20*10*10:输入通道, 500:输出通道
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)  # batch_size
        x = self.conv1(x)  # 输入: batch*1*28*28, 输出: batch*10*24*24 （28-5+1）
        x = F.relu(x)  # 保持shape不变，输出：batch*10*24*24
        x = F.max_pool2d(x, 2, 2)  # 输入：batch*10*24*24, 输出: batch*10*12*12
        x = self.conv2(x)  # 输入: batch*10*12*12, 输出: batch*20*10*10
        x = F.relu(x)  # 保持shape不变，输出：batch*20*10*10
        x = x.view(input_size, -1)  # 拉平，-1 自动计算维度， 20*10*10=2000
        x = self.fc1(x)  # 输入: batch*2000, 输出: batch*500
        x = F.relu(x)  # 保持shape不变，输出: batch*500
        x = self.fc2(x)  # 输入: batch*500, 输出: 10
        output = F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率值

        return output


# 定义优化器
model = Digit().to(DEVICE)
optimizer = optim.Adam(model.parameters())


# 定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到device上
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))


# 定义测试方法
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    # 不会进行计算梯度，也不会进行反向传播
    with torch.no_grad():
        for data, target in test_loader:
            # 部署到device上
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最大的下标
            pred = output.max(1, keepdim=True)[1]
            # pred = torch.max(output, dim=1)
            # pred = output.argmax(dim=1)
            # 累计正确的值
            correct += pred.eq(target.view_as(pred)).sum().item()
        # 计算平均loss
        test_loss /= len(test_loader.dataset)
        print("Test -- Average Loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss,
                                                                          correct / len(test_loader.dataset) * 100.0))


# 9 调用方法(7和8)
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)
