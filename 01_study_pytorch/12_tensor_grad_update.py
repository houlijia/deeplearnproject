import torch


def tensor_01():
    # 生成初始化w
    w = torch.tensor(5., requires_grad=True)
    # 训练参数
    lr = 0.01
    epoch = 5
    for i in range(epoch):
        # 生成损失函数
        loss = 3 * w ** 2 + 2 * w - 5
        # 梯度清零
        if w.grad is not None:
            w.grad.zero_()
        # 反向传播，求当前w的导数值：梯度值，斜率
        loss.backward()
        # 当前斜率
        print(f"---------{i + 1}----------")
        print(f"w.grad= {w.grad}")
        print(f"w.grad.data = {w.grad.data}")
        # 更新梯度
        # w的tensor不能修改 避免w变成新的数据 应修改w.data
        w.data = w.data - lr * w.grad.data
        print(f"w = {w}")
    # 查看训练后的w的值
    print(f"---------finally----------")
    print(f"w.data = {w.data}")


def tensor_02():
    # 生成初始化w
    w = torch.tensor([10., 20., 30.], requires_grad=True)
    # 训练参数
    lr = 0.01
    epoch = 5
    for i in range(epoch):
        # 定义损失函数
        loss = 3 * w ** 2 + 2 * w - 5
        loss = loss.sum()  # 设置一个复合函数
        # 梯度清零
        if w.grad is not None:
            w.grad.zero_()
        # 反向传播 ，求当前w的导数值：梯度值，斜率
        loss.backward()
        print(f"\n--------{i + 1}----------")
        # 当前斜率
        print(f"当前斜率：{w.grad}")
        # 更新梯度
        w.data = w.data - lr * w.grad.data
        print(f"w = {w.data}")

    print("-------finally--------")
    print(f"w = {w.data}")
    # save data
    torch.save(w.data, './12_tensor_grad_update-w_data.pth')


def load_data():
    w = torch.load('./12_tensor_grad_update-w_data.pth', map_location='mps')
    print(f"w.device: {w.device}")


if __name__ == '__main__':
    # tensor_01()
    tensor_02()
    load_data()
