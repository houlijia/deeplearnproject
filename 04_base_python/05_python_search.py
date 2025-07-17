import re

text = "Python is awesome"
result1 = re.search(r'is', text)
print(result1)  # 匹配成功，返回匹配对象

text = "Python is awesome"
result2 = re.match(r'Python', text)  # 匹配成功
print(result2)
result3 = re.match(r'is', text)     # 匹配失败，返回None
print(result3)
