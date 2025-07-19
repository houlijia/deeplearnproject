
# 用法1、
print("---- 用法1、")
def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1

counter = count_up_to(5)
for i in counter:
    print(i)

# 用法2、逐行处理大文件而不占用过多内存
print("---- 用法2、逐行处理大文件而不占用过多内存")
def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# 逐行处理大文件而不占用过多内存
for line in read_large_file('./README.txt'):
    print(line)

# 用法3、可以无限生成数字而不耗尽内存
print("---- 用法3、可以无限生成数字而不耗尽内存")
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1
# 可以无限生成数字而不耗尽内存
for i in infinite_sequence():
    if i > 10:
        break
    print(i)


# 用法4、yield还可以用于协程，实现双向通信
print("---- 用法4、yield还可以用于协程，实现双向通信")
def coroutine():
    print("启动协程")
    while True:
        value = yield
        print(f"接收到的值为: {value}")

cor = coroutine()
next(cor)
cor.send(10)
cor.send(20)
