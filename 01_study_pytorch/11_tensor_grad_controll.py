import torch


# 正常情况下
def tensortest01():
    x = torch.tensor(10.5, requires_grad=True)
    print(f"tensortest01()_x: {x.requires_grad}")  # True

    # 默认y的requires_grad=True
    y = x ** 2 + 2 * x + 3
    print(f"tensortest01()_y: {y.requires_grad}")  # True


def tensortest02():
    x = torch.tensor(10.5, requires_grad=True, dtype=torch.float64)
    # 关闭梯度计算，使用with进行上下文管理
    with torch.no_grad():
        y = x ** 2 + 2 * x + 3
        print(f"tensortest02()_y: {y.requires_grad}")   # False


# 使用装饰器torch.no_grad()关闭梯度计算
@torch.no_grad()
def tensortest03():
    x = torch.tensor(10.5, requires_grad=True,dtype=torch.float64)
    y = x ** 2 + 2 * x + 3
    y01 = y.sum()
    print(f"tensortest03()_y: {y01.requires_grad}")  # True


# 全局设置set_grad_enabled(False)关闭梯度计算
def tensortest04():
    x = torch.tensor([10.5, 20.5], requires_grad=True, dtype=torch.float64)
    torch.set_grad_enabled(False)
    y = x ** 2 + 2 * x + 3
    y01 = y.sum()
    print(f"tensortest04()_y: {y01.requires_grad}")
    torch.set_grad_enabled(True)


# 累计梯度
def tensortest05():
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    y = x**2 + x*5 + 10
    y = y.sum()
    y.backward()
    print(f"y1_x.grad= {x.grad}")

    y = x**3 + 10*x - 10
    y = y.sum()
    y.backward()
    print(f"y2_x.grad= {x.grad}")


# 累计梯度02
def tensortest06():
    x = torch.tensor(3, requires_grad=True, dtype=torch.float64)
    for _ in range(4):
        y = x**2 + x*3 - 10
        y.backward()
        print(f"for range x.grad= {x.grad}")
        # 梯度置0 ，若不置0，梯度累加，会导致梯度爆炸
        if x.grad is not None:
            x.grad.zero_()


if __name__ == '__main__':
    tensortest01()
    tensortest02()
    tensortest03()
    tensortest04()
    tensortest05()
    tensortest06()
