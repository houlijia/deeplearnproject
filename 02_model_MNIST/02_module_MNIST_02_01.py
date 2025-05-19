import cv2
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
print(torch.__version__)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=transforms.ToTensor(),  # 将数据转换成tensor的格式
    download=True,
)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)  # shuffle 表示打乱
device = ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"device is:{device}")
# 加载已经训练好的模型文件
cnn = torch.load('model/mnist_model_02.pkl')
cnn = cnn.to('cpu')
# 损失函数
loss_function = nn.CrossEntropyLoss()
loss_test = rightValue = 0
for index, (images, labels) in enumerate(test_loader):
    image = images.to(device)
    labels = labels.to(device)
    # 前向传播
    outputs = cnn(images)
    _, pred = outputs.max(1)
    loss_test += loss_function(outputs, labels)
    images = images.to('cpu').numpy()
    labels = labels.to('cpu').numpy()
    pred = pred.to('cpu').numpy()

    for idx in range(images.shape[0]):
        im_data = image[idx]
        imdata = im_data.transpose(1, 2, 0)
        im_label = labels[idx]
        im_pred = pred[idx]
        print(f"预测值为{im_pred}")
        print(f"真实值为{im_label}")
        cv2.imshow("now image", im_data)
        cv2.waitKey(0)

print("loss为{},准确率为{}".format(loss_test, rightValue / len(test_data)))

