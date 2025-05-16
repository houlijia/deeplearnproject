import torch

data = torch.randn(3, 4)
print(f"data:\n{data}\n")

# 向下取整(往小取 返回不大于每个元素的最大整数)
floor_tensor = torch.floor(data)
print(f"floor_tensor:\n{floor_tensor}\n")

# 向上取整(往大取 返回不小于每个元素的最小整数)
ceil_tensor = torch.ceil(data)
print(f"ceil_tensor:\n{ceil_tensor}\n")

# 四舍五入 与python round()一致 四舍六入五看齐(看个位奇偶，奇进偶舍)
round_tensor = torch.round(data)
print(f"round_tensor:\n{round_tensor}\n")

# 取绝对值
abs_tensor = torch.abs(data)
print(f"abs_tensor:\n{abs_tensor}\n")

# 取平方
square_tensor = torch.square(data)
print(f"square_tensor:\n{square_tensor}\n")

# 取平方根
sqrt_tensor = torch.sqrt(data)
print(f"sqrt_tensor:\n{sqrt_tensor}\n")

# 取对数
log_tensor = torch.log(data)
print(f"log_tensor:\n{log_tensor}\n")

# 截断
trunc_tensor = torch.trunc(data)
print(f"trunc_tensor:\n{trunc_tensor}\n")

# 取余
fomd_tensor = torch.fmod(data, 2)
print(f"fomd_tensor:\n{fomd_tensor}\n")

# Trigonometric function （三角函数）
print("----------------Trigonometric function (三角函数)---------------")
sin_tensor = torch.sin(data)
print(f"sin_tensor:\n{sin_tensor}\n")

cos_tensor = torch.cos(data)
print(f"cos_tensor:\n{cos_tensor}\n")

tan_tensor = torch.tan(data)
print(f"tan_tensor:\n{tan_tensor}\n")

asin_tensor = torch.asin(data)
print(f"asin_tensor:\n{asin_tensor}\n")

acos_tensor = torch.acos(data)
print(f"acos_tensor:\n{acos_tensor}\n")

atan_tensor = torch.atan(data)
print(f"atan_tensor:\n{atan_tensor}\n")


print("----------------Statistical function (统计函数)------------------")
print(f"data:\n{data}\n")
# 均值
mean_tensor = torch.mean(data)
print(f"mean_tensor:\n{mean_tensor}\n")
# 标准差
std_tensor = torch.std(data)
print(f"std_tensor:\n{std_tensor}\n")
# 中位数
median_tensor = torch.median(data)
print(f"median_tensor:\n{median_tensor}\n")
# 众数
mode_tensor = torch.mode(data)
print(f"mode_tensor:\n{mode_tensor}\n")
# 最大值
max_tensor = torch.max(data)
print(f"max_tensor:\n{max_tensor}\n")
# 最小值
min_tensor = torch.min(data)
print(f"min_tensor:\n{min_tensor}\n")
# 累加
sum_tensor = torch.sum(data)
print(f"sum_tensor:\n{sum_tensor}\n")
# 累乘
prod_tensor = torch.prod(data)
print(f"prod_tensor:\n{prod_tensor}\n")

# 排序
sort_tensor, sort_indices = torch.sort(data)
print(f"data:\n{data}\n")
print(f"sort_tensor:\n{sort_tensor}\n")
print(f"sort_indices:\n{sort_indices}\n")

# histc 计算张量的直方图
histc_tensor = torch.histc(data, bins=5)
print(f"histc_tensor:\n{histc_tensor}\n")

# unique 计算张量的唯一值和其对应的索引
data = torch.tensor([1, 2, 3, 2, 4, 1, 5])
unique_tensor, unique_indices = torch.unique(data, sorted=True, return_inverse=True)
print(f"unique_tensor:\n{unique_tensor}\n")
print(f"unique_indices:\n{unique_indices}\n")

# bincount 计算张量中每个元素出现的次数
data = torch.tensor([1, 2, 3, 2, 4, 1, 5])
bincount_tensor = torch.bincount(data)  # 会考虑到0的存在
print(f"bincount_tensor:\n{bincount_tensor}\n")

# cumsum 计算张量的累加
cumsum_tensor = torch.cumsum(data, dim=0)
print(f"cumsum_tensor:\n{cumsum_tensor}\n")

print("----------------save and load tensor-------------")
save_tensor = torch.randn(3, 4)
torch.save(save_tensor, "./save_tensor.pt")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"availabledevice is: {device}")
load_tensor = torch.load("./save_tensor.pt", map_location=device)
print(f"save_tensor:\n{save_tensor}\n")
print(f"load_tensor:\n{load_tensor}\n")
print(f"load_tensor.device:\n{load_tensor.device}\n")


