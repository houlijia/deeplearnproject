from contextlib import contextmanager


# 0、上下文管理器通过 with 语句使用，最常见的例子是文件操作：
with open('./README.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        print(line)


# 自定义实现上下文管理器
# 方式1: contextlib模块
print("---- 方式1: contextlib模块 ----")
@contextmanager
def my_context_manager():
    print("进入上下文")
    try:
        yield "资源对象"  # 这个是as 后面的值
    except Exception as e:
        print(f"发生异常：{e}")
        raise  # 可以选择重新抛出异常
    finally:
        print("退出上下文")


# 使用示例
with my_context_manager() as mycm:
    print(f"正在使用资源: {mycm}")


# 方式2、使用类实现
# 需要实现 __enter__ 和 __exit__ 方法
print("---- 方式2、使用类实现 ----")
class MyContextManager:
    def __enter__(self):
        print("进入上下文")
        # 返回资源对象
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("退出上下文")
        # 处理异常，如果返回 Ture 则异常被抑制
        if exc_type:
            print(f"发生异常: {exc_type}")

        # 不需要返回任何值或返回 None时, 异常会继续传播
        return False


# 使用示例
with MyContextManager() as mcm:
    print("在上下文中")


# 多个上下文管理器

# 3、可以同时使用多个上下文管理器：
print("---- 3、可以同时使用多个上下文管理器 ----")
with open('README.txt', 'r') as fin, open('output.txt', 'w') as fout:
    for line in fin:
        fout.write(line.upper())
print("多个上下文管理器使用完毕")
