import torch

# 创建标量：要计算梯度则必须为浮点类型
print("--------创建标量、计算梯度--------")
x = torch.tensor(3, requires_grad=True, dtype=torch.float64)
# 定义函数
y = x**2 + x*2 + 1
y.backward()
# 计算梯度
print(f"x={x.item()}, y={y.item()}, x.grad={x.grad.item()}")

# 创建向量
print("--------创建向量、计算梯度--------")
x = torch.tensor([1, 2, 3], requires_grad=True, dtype=torch.float64)
# 定义函数
y = x**2 + x*2 + 1
print(f"x={x}, y={y}")

y = y.sum()  # y必须是一个标量才能用这个标量对x求导 设置一个复合函数
print(f"y.sum()={y}")

y.backward()
# 计算梯度
print(x.grad)

# 多标量梯度计算
print("--------多标量梯度计算--------")
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)

y = x1**2 + x2**2
y.backward()

print(f"x1.grad={x1.grad}, x2.grad={x2.grad}")

# 多向量梯度计算
print("--------多向量梯度计算--------")
x1 = torch.tensor([1, 2, 3], requires_grad=True, dtype=torch.float64)
x2 = torch.tensor([4, 5, 6], requires_grad=True, dtype=torch.float64)

y = x1**2 + x2**2
y = y.sum()
y.backward()
print(f"x1.grad={x1.grad}, x2.grad={x2.grad}")

# 矩阵梯度计算
print("--------矩阵梯度计算--------")
x = torch.tensor([[1, 2], [3, 4]], requires_grad=True, dtype=torch.float64)

y = x**2
y = y.sum()
y.backward()
print(f"x.grad={x.grad}")

# 矩阵的行列式梯度计算
print("--------矩阵的行列式梯度计算--------")
x = torch.tensor([[1, 2], [3, 4]], requires_grad=True, dtype=torch.float64)

y = torch.det(x)
y.backward()
print(f"x.grad={x.grad}")

# 矩阵的逆梯度计算
print("--------矩阵的逆梯度计算--------")
x = torch.tensor([[1, 2], [3, 4]], requires_grad=True, dtype=torch.float64)

y = torch.inverse(x)
y1 = y.sum()
y1.backward()
print(f"x.grad={x.grad}")

