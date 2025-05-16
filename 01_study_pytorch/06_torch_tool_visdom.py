import visdom
import numpy as np

"""
 step
 1、pip install visdom
 2、python -m visdom.server /python -m visdom.server -p 8098 , 默认端口为8097，这里使用8098
 3、open http://localhost:8098/
"""

vis = visdom.Visdom()

vis.text('Hello, world!')

vis.image(np.ones(3, 100, 100))

