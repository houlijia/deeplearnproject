from cffi import FFI

ffi = FFI()

ffi.cdef("""
    int add(int a, int b);
""")

C = ffi.dlopen('./libtestc.so')  # 确保路径正确，并且共享库已经存在

result = C.add(1, 2)
print(result)  # 输出: 3
