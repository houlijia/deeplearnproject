from mylogger.mylogger import *

# 使用type动态创建类
MyClass = type('MyClass', (), {'x': 10, 'y': 11})
# 相当于
# class Myclass:
#   x = 10
#   y = 11

obj = MyClass()
print(type(obj))
print(f"x= {obj.x}")
print(f"y= {obj.y}")

mylogger.info("---------")
print(type(10))
