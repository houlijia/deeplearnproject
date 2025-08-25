s = "hello yuan"
s2 = str("hello world")
print(s2, type(s2))

l = [1, 2, 3]
l2 = list([1, 2, 3])
print(l2, type(l2))

d = {"a": "A"}
print(dict([("a", 1), ("b", 2)]))


aaa = [1, 2, 3, 3, 4, 5, 6, 6]

print(aaa)
a = (1, 2, 3, 1)
print(a.count(1))
print(a.index(1))

def foo(x):
    print(id(x))
    x.append(5)

x = [1, 2, 3, 4]
print(id(x))
foo(x)
print(x)

