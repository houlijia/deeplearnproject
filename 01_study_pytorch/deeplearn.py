# from mxnet  import  np,npx
import torch
import numpy as np

# 检查mps是否可用
is_mps_available = torch.backends.mps.is_available()
# mps非空闲时使用cpu
device = torch.device('mps' if (torch.backends.mps.is_available()) else 'cpu')

torch_tensor = torch.tensor(np.arange(15).reshape(3, 5))
print(f"PyTorch Tensor = \n{torch_tensor}\n")

torch_tensor.zero_()
print(f"PyTorch Tensor = \n{torch_tensor}\n")

torch_tensor.add_(1)
print(f"PyTorch Tensor = \n{torch_tensor}\n")

torch_tensor.fill_(2)
print(f"PyTorch Tensor = \n{torch_tensor}\n")

# stack 为拼接函数
A = torch.tensor([[1, 2], [3, 4]])
print(f"PyTorch Tensor A= \n{A}\n")
print(f"PyTorch Tensor A.shape = {A.shape}\n")
B = torch.tensor([[5, 6], [7, 8]])
print(f"PyTorch Tensor B = \n{B}\n")
print(f"PyTorch Tensor B.shape = {B.shape}\n")
C = torch.stack([A, B], dim=0)
print(f"PyTorch Tensor = \n{C}\n")
print(f"PyTorch Tensor C.shape = {C.shape}\n")

A1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
B1 = torch.tensor([[7, 8, 9], [10, 11, 12]])
print(f"PyTorch Tensor A1= \n{A1}\n")
print(f"PyTorch Tensor A1.shape = {A1.shape}\n")  # (2, 3)
C1 = torch.stack([A1, B1], dim=1)
print(f"PyTorch Tensor = \n{C1}\n")
print(f"PyTorch Tensor C1.shape = {C1.shape}\n")  # (2, 2, 3)


# cat 与 stack 类似，但可以指定维度
D = torch.cat([A, B], dim=1)
print(f"PyTorch Tensor cat D= \n{D}\n")

A = torch.tensor([[1, 2], [3, 4]])
# unsqueeze 增加维度
E = torch.unsqueeze(A, dim=0)
print(f"PyTorch Tensor unsqueeze E= \n{E}\n")
print(f"PyTorch Tensor unsqueeze E.shape = {E.shape}\n")

# squeeze 减少维度
F = torch.squeeze(E, dim=0)
print(f"PyTorch Tensor squeeze F= \n{F}\n")

# permute 改变维度顺序
G = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"PyTorch Tensor G= \n{G}\n")
H = G.permute(1, 0)
print(f"PyTorch Tensor permute H= \n{H}\n")



