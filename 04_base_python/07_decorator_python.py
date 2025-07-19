# 1、decorator 基本使用
print("----1、decorator 基本使用----")


def my_decorator(func):
    def wrapper():
        print("在函数执行前执行")
        func()
        print("在函数执行后执行")

    return wrapper


@my_decorator
def say_hello():
    print("你好")


say_hello()

# 2、带参数的函数装饰器
print("----2、带参数的函数装饰器----")


def great_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"准备调用{func.__name__}()函数...")
        result = func(*args, **kwargs)
        print(f"在收尾阶段调用{func.__name__}()函数...")
        return result

    return wrapper


@great_decorator
def greet(name):
    print(f"你好, {name}")


greet("张三")

# 3、带参数的装饰器
print("----3、带参数的装饰器----")


def repeat(runtimes):
    def decorator_repeat(func):
        def wrapper(*args, **kwargs):
            print("准备工作...")
            for _ in range(runtimes):
                result = func(*args, **kwargs)
            print("后处理阶段...")
            return result

        return wrapper

    return decorator_repeat


@repeat(runtimes=3)
def greet(name):
    print(f"你好, {name}")


greet("张三")

# 4、类装饰器
print("---- 4、类装饰器----")


class CountCalls:
    def __init__(self, func):
        self.func = func
        self.num_calls = 0

    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        print(f"调用次数：{self.num_calls}")
        return self.func(*args, **kwargs)


@CountCalls
def say_hello_class():
    print("你好, Hello")


say_hello_class()
say_hello_class()

# 5、多个装饰器叠加
print("---- 5、多个装饰器叠加----")


def decorator1(func):
    def wrapper():
        print("装饰器1 - 前")
        func()
        print("装饰器1 - 后")

    return wrapper


def decorator2(func):
    def wrapper():
        print("装饰器2 - 前")
        func()
        print("装饰器2 - 后")

    return wrapper


@decorator2
@decorator1
def my_function():
    print("原始函数")


my_function()

# 6、内置装饰器
print("---- 6、内置装饰器----")


class MyClass:
    @property  # - 将方法转换为属性
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @classmethod  # - 定义类方法
    def class_method(cls):
        print(f"这是一个类方法，类名为 {cls.__name__}")

    @staticmethod  # - 定义静态方法
    def static_method():
        print("这是一个静态方法")


class1 = MyClass
class1.class_method()
print(f"class1.value: {class1}")

# 7、计数、日志等简单功能，函数装饰器通
print("---- 7、计数、日志等简单功能，函数装饰器通 ----")
from functools import wraps


def count_calls(func):
    num_calls = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal num_calls
        num_calls += 1
        print(f"调用次数：{num_calls}")
        return func(*args, **kwargs)

    return wrapper


@count_calls
def say_hello_func():
    print("你好, Hello")


say_hello_func()
say_hello_func()
