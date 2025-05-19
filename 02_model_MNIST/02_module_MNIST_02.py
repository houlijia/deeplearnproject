import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 3、定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            #  1、卷积操作
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  #
            # 2、归一化操作 BN层
            nn.BatchNorm2d(32),
            # 3、激活层 ReLu层
            nn.ReLU(),
            # 4、最大池化
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(in_features=14 * 14 * 32, out_features=10)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        out = self.conv(x)
        # 将图像展开为1维
        # 输入的张量（n, c, h, w）
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


# 1、数据加载
train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=transforms.ToTensor(),  # 将数据转换成tensor的格式
    download=True,
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=transforms.ToTensor(),  # 将数据转换成tensor的格式
    download=True,
)

# print(f"---train data--- is:\n {train_data}")
# print(f"---test data--- is:\n {test_data}")


# 2、数据加载（分批）
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)  # shuffle 表示打乱
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)  # shuffle 表示打乱

# print(train_loader)
# print(test_loader)
#
cnn = CNN()
# 关键设置：禁用NNPack并启用MPS
torch.backends.nnpack.enabled = False
torch.backends.mps.enable = True
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'
# cnn = cnn.to(device).to(torch.float32)

# 损失函数
loss_function = nn.CrossEntropyLoss()

# 优化函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

for epoch in range(10):
    for index, (images, labels) in enumerate(train_loader):
        # print(index)
        # print(images)
        # print(labels)
        image = images.to(device)
        labels = labels.to(device)
        # 前向传播

        outputs = cnn(images)

        loss = loss_function(outputs, labels)

        # 梯度置0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("当前为第{}轮，批次为{}/{}, loss为{}".format(epoch + 1, index + 1, len(train_data) // 64, loss.item()))
        # break  # 打印一次就break

    loss_test = rightValue = 0
    for index2, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = cnn(images).to(device)
        # print(outputs)
        # print(outputs.size())
        # print(labels)

        loss_test += loss_function(outputs, labels)
        _, pred = outputs.max(1)
        # print(pred)
        # print(pred.eq(labels))  # 将张量中的每一个元素进行对比，如果相等，返回true，如果不相等，对应为false，返回一个张量
        rightValue += (pred.eq(labels).sum().item())
        print("当前为第{}轮验证测试集，当前批次为{}/{},loss为{},准确率为{}".format(epoch+1, index2+1, len(test_data)//64, loss_test, rightValue/len(test_data)))

torch.save(cnn, 'model/mnist_model_02.pkl')
