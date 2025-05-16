import numpy as np
import torch

if torch.backends.mps.is_available():
    print("MPS is available!")
else:
    print("MPS is not available.")

# 当前安装的 PyTorch 库的版本
print(f"PyTorch version: {torch.__version__}")

single_value = torch.tensor(42)
single_value_item = single_value.item()
print(f"single_value_item: {type(single_value_item)}")
print(f"single_value: {single_value.item()}")
# 创建一个 2x3 矩阵
a01 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
print(f"a01 value:\n{a01}")
print(f"a01_T: {a01.T}")
# tensor attribute
print(a01.shape)
print(a01.dtype)
print(f"a01.requires_grad: {a01.requires_grad}")
print(f"a01.size(): {a01.size()}")
print(f"a01.dim(): {a01.dim()}")
print(f"a01.numel(): {a01.numel()}")
print(f"a01.is_cuda: {a01.is_cuda}")
print(f"a01.device: {a01.device}")
print(f"a01.dtype: {a01.dtype}")
if torch.backends.mps.is_available():
    a01 = a01.to('mps')
    print(f"a01.is_mps: {a01.is_mps}")
    print(f"a01.is_cuda: {a01.is_cuda}")
    print(f"a01.device: {a01.device}")   # 输出：mps:0


zero_tensor = torch.zeros(3, 3)
print(f"zero_tensor value:\n{zero_tensor}")
ones_tensor = torch.ones(3, 3)
print(f"ones_tensor:{ones_tensor}")

# 随机生成一个 3x3 矩阵
rand_tensor = torch.rand(3, 3)
print(f"rand_tensor: {rand_tensor}")

# 利用现有矩阵创建新矩阵
new_tensor = zero_tensor + rand_tensor
print(f"new_tensor: {new_tensor}")

arange_tensor = torch.arange(1, 10, 2)  # 生成从 1 到 9 步长为 2 的序列
print(f"arange_tensor: {arange_tensor}")
linspace_tensor = torch.linspace(1, 10, 5)  # 生成从 1 到 10 均匀分成 5 个部分的序列
torch.linspace(1, 10, 5, out=zero_tensor)  # 将结果输出到 linspace_tensor
print(f"zero_tensor: {zero_tensor}")
print(f"linspace_tensor: {linspace_tensor}")
long_tensor = torch.logspace(1, 10, 5)  # 生成从 10 的 10 倍到 1 的 5 个部分的对数序列
print(f"long_tensor: {long_tensor}")

eye_tensor = torch.eye(3)  # 生成 3x3 单位矩阵
print(f"eye_tensor: {eye_tensor}")

# 假设你有一个在 MPS 上的张量
mps_tensor = torch.randn(3, 3, device='mps')  # 在 MPS 设备上创建张量
# 将张量从 MPS 转移到 CPU
cpu_tensor = mps_tensor.to('cpu')
print(cpu_tensor.device)  # 输出: cpu

print(f"a01.device: {a01.device}")
a01 = a01.cpu()
print(f"a01.device: {a01.device}")
print(f"a01.to('cpu'): {a01}")

numpy_array = a01.numpy()  # 将 PyTorch 张量转换为 NumPy 数组
a02 = torch.ones((2, 3))  # 创建一个 NumPy 数组
print(f"a02: {a02}")
print(f"Pytorch to numpy_array: {numpy_array}")
torch_array = torch.from_numpy(numpy_array)
a03 = torch.from_numpy(numpy_array)  # 将 NumPy 数组转换为 PyTorch 张量
print(f"numpy_array to torch_array: {a03}")

print("--------------------------------------------------------")
# 其他维度的创建
tensor_2d = torch.tensor([[1, 1, 1], [21, 21, 21], [31, 31, 31]])
tensor_2d02 = tensor_2d*2
tensor_2d03 = tensor_2d*3
# tensor_2d03 = tensor_2d*3

# stack堆叠
print("----------------- stack 拼接-----------------------")
print(f"tensor_2d.shape: {tensor_2d.shape}")
tensor_3d = torch.stack([tensor_2d, tensor_2d02, tensor_2d03], dim=2)
print(f"tensor_3d: {tensor_3d}")
print(f"tensor_3d.shape: {tensor_3d.shape}")
print(f"tensor_3d[:, :, 0]: {tensor_3d[:, :, 0]}")

x = torch.tensor([[1, 2, 3],
        [4, 5, 6]])
y = torch.tensor([[11, 22, 33],
        [44, 55, 66]])
tensor_x_2d = torch.stack((x, y), dim=2)
print(f"tensor_x_2d: {tensor_x_2d}")
print(f"tensor_x_2d.shape: {tensor_x_2d.shape}")
print(f"tensor_x_2d[:, :, 0]: {tensor_x_2d[:, :, 1]}")

# cat拼接
print("----------------- cat 拼接-----------------------")
ones_tensor_01 = torch.ones(2, 3)
ones_tensor_02 = torch.ones(2, 3)*2
cat_tensor = torch.cat((ones_tensor_01, ones_tensor_02), dim=0)
print(f"cat_tensor_dim0: {cat_tensor}")
cat_tensor = torch.cat((ones_tensor_01, ones_tensor_02), dim=1)
print(f"cat_tensor_dim1: {cat_tensor}")

