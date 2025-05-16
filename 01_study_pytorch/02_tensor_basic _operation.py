import torch

# Create tensors
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Add tensors
z = x + y
print(f"x + y = {z}")

# Multiply tensors
z = x * y
print(f"x * y = {z}")

# Dot product
z = torch.dot(x, y)
print(f"Dot product of x and y = {z}")  # Output: 32 = 1*4 + 2*5 + 3*6

# sum
z = torch.sum(x)
print(f"Sum of x = {z}")  # Output: 6 = 1 + 2 + 3

# mean
x = torch.tensor([1, 2, 3], dtype=torch.float32)
z = torch.mean(x)
print(f"Mean of x = {z}")  # Output: 2.0 = (1 + 2 + 3) / 3

# max
z = torch.max(x)
print(f"Max of x = {z}")  # Output: 3

# min
z = torch.min(x)
print(f"Min of x = {z}")  # Output: 1

# argmix
z = torch.argmax(x)
print(f"Argmax of x = {z}")  # Output: 2 (index of the maximum value)

# argmin
z = torch.argmin(x)
print(f"Argmin of x = {z}")  # Output: 0 (index of the minimum value)
# transpose
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
z = x.transpose(0, 1)
print(f"Transpose of x = {z}")  # Output: tensor([[1, 4], [2, 5], [3, 6]])

# softmaxhand 函数将一个 n 维的实值向量转换成概率分布，使得每一个元素的值介于 0 到 1 之间，并且所有元素的总和等于 1。softmax 函数的公式如下：
# softmax(x_i) = e^(x_i) / sum_j(e^(x_j))
# 其中，x_i 是输入向量中的第 i 个元素，e^(x_i) 是指数函数。
# 为了计算 softmax 函数，我们需要对输入向量进行归一化处理，使得所有元素的总和等于 1。
# 我们可以使用 torch.softmax 函数来实现 softmax 函数。
softmax_input = torch.tensor([1, 2, 3], dtype=torch.float32)
softmax_output = torch.softmax(softmax_input, dim=0)
print(f"Softmax of {softmax_input} = {softmax_output}")  # Output: tensor([0.0900, 0.2447, 0.6652])

# view
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
z = x.view(6)  # reshape 改变张量的形状（不改变数据）
print(f"View of x = {z}")  # Output: tensor([1, 2, 3, 4, 5, 6])
z = x.view(3, 2)  # 第二个维度由原来的 2 变成了 3
print(f"View of x = {z}")  # Output: tensor([[1, 2], [3, 4], [5, 6]])
z = x.view(-1, 2)  # 第一个维度由原来的 2 变成了 -1，表示由其他维度推断出来的维度
print(f"View of x = {z}")  # Output: tensor([[1, 2, 3], [4, 5, 6]])

# reshape, 类似于 view，但更灵活
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
z = x.reshape(6)  # reshape 改变张量的形状（不改变数据）
print(f"Reshape of x = {z}")  # Output: tensor([1, 2, 3, 4, 5, 6])
z = x.reshape(3, 2)  # 第二个维度由原来的 2 变成了 3
print(f"Reshape of x = {z}")  # Output: tensor([[1, 2], [3, 4], [5, 6]])
z = x.reshape(-1, 2)  # 第一个维度由原来的 2 变成了 -1，表示由其他维度推断出来的维度
print(f"Reshape of x = {z}")  # Output: tensor([[1, 2, 3], [4, 5, 6]])

# permute, 改变张量的维度顺序
x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(f"Shape of x = {x.shape}")
print(f"x = {x}")
z = x.permute(2, 0, 1)  # 交换三个维度的位置
print(f"Shape of z = {z.shape}")
print(f"Permute of x = {z}")  # Output: tensor([[[ 1,  7], [ 4, 10]],[[ 2,  8], [ 5, 11]],[[ 3,  9], [ 6, 12]]])
# z = x.permute(1, 0)  # 交换两个维度的位置
# print(f"Permute of x = {z}")  # Output: tensor([[1, 4], [2, 5], [3, 6]])

# unsqueeze, 在指定维度上增加维度
print("--------unsqueeze--------")
unsqueeze_x = torch.tensor([1, 2, 3])
print(f"unsqueeze_x = {unsqueeze_x}")
print(f"Shape of unsqueeze_x = {unsqueeze_x.shape}")
unsqueeze_z_0 = unsqueeze_x.unsqueeze(0)  # 在 0 维增加一个维度
print(f"Shape of unsqueeze_z_0 = {unsqueeze_z_0.shape}")
print(f"Unsqueeze of x = {unsqueeze_z_0}")  # Output: tensor([[1, 2, 3]])
unsqueeze_z_1 = unsqueeze_x.unsqueeze(1)  # 在 1 维增加一个维度
print(f"Shape of unsqueeze_z_1 = {unsqueeze_z_1.shape}")
print(f"Unsqueeze of unsqueeze_z_1 = {unsqueeze_z_1}")  # Output: tensor([[1], [2], [3]])

# squeeze, 去掉指定维度上的维度
print("--------squeeze--------")
squeeze_x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(f"Shape of squeeze_x = {squeeze_x.shape}")
print(f"squeeze_x = {squeeze_x}")
squeeze_z_0 = squeeze_x.squeeze(0)  # 去掉 0 维上的维度
print(f"Shape of squeeze_z_0 = {squeeze_z_0.shape}")
print(f"Squeeze of squeeze_z_0 = {squeeze_z_0}")  # Output: tensor([[1, 2, 3], [4, 5, 6]])

# 示例1：第0维的维度大小为1，squeeze(0)会移除该维度
x1 = torch.randn(1, 3, 3)  # 形状：[1, 3, 3]
y1 = x1.squeeze(0)         # 形状：[3, 3]
print(y1.shape)  # 输出：torch.Size([3, 3])

# 示例2：第0维的维度大小不为1，squeeze(0)不会改变张量
x2 = torch.randn(2, 3, 3)  # 形状：[2, 3, 3]
y2 = x2.squeeze(0)         # 形状：[2, 3, 3]
print(y2.shape)  # 输出：torch.Size([2, 3, 3])

# 示例3：使用squeeze()（不带参数）会移除所有维度大小为1的维度
x3 = torch.randn(1, 3, 1, 2)  # 形状：[1, 3, 1, 2]
y3 = x3.squeeze()             # 形状：[3, 2]
print(y3.shape)  # 输出：torch.Size([3, 2])

