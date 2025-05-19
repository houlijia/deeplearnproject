import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义超参数
BATCH_SIZE = 512
EPOCHS = 20  # 总共需要训练20次
DEVICE = ('mps' if torch.backends.mps.is_available() else 'cpu')

pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 将图片进行归一化，降低模型复杂程度
])

train_data = datasets.MNIST(root="data", train=True, download=True, transform=pipeline)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=pipeline)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # conv2d的第一个参数为输入通道数, 第二个为输出通道数, 第三个参数为卷积核大小
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        # 全连接层的第一个参数指输入通道数, 第二个指输出通道数
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)  # 输入通道是500, 输出通道是10, 即 10分类

    def forward(self, x):
        in_size = x.size(0)  # 本例中in_size = 512, 也就是BATCH_SIZE的值, 输入的x可以看成 512*1*28*28的张量
        out = self.conv1(x)  # batch*1*28*28 -> batch*10*24*24 (28*28图片经过5*5的卷积，输出为24*24)
        out = F.relu(out)    # batch*10*24*24 , 激活函数ReLU 不改变形状
        out = F.max_pool2d(out, 2, 2)  # batch*10*24*24 -> batch*10*12*12 (2*2池化层会减半)
        out = self.conv2(out)  # batch*10*12*12  -> batch*20*10*10 (再卷积一次，核的大小是3)
        out = F.relu(out)
        out = out.view(in_size, -1)  # batch*20*10*10 ->batch*2000 (out的第二维是-1，说明是自动推断，本例中第二维是20*10*10)
        out = self.fc1(out)  # batch*2000 -> batch*500
        out = F.relu(out)  # batch*500
        out = self.fc2(out)  # batch*500  -> batch *10
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
