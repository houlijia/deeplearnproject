# 闭包示例代码


def outer_function(x):
    # 外部函数的变量
    message = "Hello"

    def inner_function(y):
        # 内部函数访问外部函数的变量（自由变量）
        return f"{message}, {x} + {y} = {x + y}"

    # 返回内部函数（闭包）
    return inner_function


# 创建闭包
closure1 = outer_function(10)
closure2 = outer_function(20)

# 调用闭包函数
print(closure1(5))  # 输出: Hello, 10 + 5 = 15
print(closure2(3))  # 输出: Hello, 20 + 3 = 23

# 检查闭包属性
print(closure1.__closure__)  # 输出: (<cell at ...: int object at ...>, <cell at ...: str object at ...>)
print(closure1.__closure__[0].cell_contents)  # 输出: 10
print(closure1.__closure__[1].cell_contents)  # 输出: Hello