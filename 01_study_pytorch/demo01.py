import numpy as np
import matplotlib.pyplot as plt

# 4D数据 (3空间维度 + 1颜色维度)
data = np.random.rand(5, 5, 5, 3)  # 最后维度表示RGB颜色

# 创建3D散点图，颜色表示第四维
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x, y, z = np.indices(data.shape[:3])
x = x.flatten()
y = y.flatten()
z = z.flatten()
colors = data.reshape(-1, 3)  # 展平颜色数据

ax.scatter(x, y, z, c=colors, s=50, alpha=0.7)
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')
plt.title('4D数据可视化 (RGB颜色表示第四维)')
plt.show()