from ctypes import CDLL, c_int

# 加载共享库
lib = CDLL('./libtestc.so')
# 设置函数参数类型
lib.add.argtypes = [c_int, c_int]
# 设置返回类型
lib.add.restype = c_int

# 调用函数
result = lib.add(1, 2)
print(result)  # 输出: 3
