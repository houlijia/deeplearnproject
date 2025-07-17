import gc

# 查看当前各代阈值
print(gc.get_threshold())  # 默认 (700, 10, 10)

# 手动触发垃圾回收
gc.collect()

# 禁用/启用 GC
gc.disable()
gc.enable()
