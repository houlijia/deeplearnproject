import copy

print("---------------浅copy-------------------------")
original_list = [1, 2, [3, 4]]
copy_value = copy.copy(original_list)
print(copy_value)
print(copy_value is original_list)

# 修改浅拷贝后的列表
copy_value[0] = 100  # 不会影响原列表
print(f"修改copy_value[0] = 100")
print(f"修改copy_value后，original_list:{original_list}")  # [1, 2, [3, 4]]
print(f"修改copy_value后，copy_value:{copy_value}")  # [100, 2, [3, 4]]
print("")
original_list[2][0] = 300  # 会影响原列表中的子列表
print(f"修改original_list[2][0] = 300")
print(f"修改original_list后，original_list:{original_list}")  # [1, 2, [300, 4]]
print(f"修改original_list后，copy_value:{copy_value}")  # [100, 2, [300, 4]]
print("浅copy的特点是:修改原对象的值不会影响copy对象的值；但修改copy对象的值会影响原对象的值")
print("浅拷贝创建一个新对象，但不会递归复制对象内部的子对象，而是直接引用原对象中的子对象")
print("")

print("---------------deepcopy-----------------")
original_list = [1, 2, [3, 4]]
deep_copied_list = copy.deepcopy(original_list)
print(f"deepcopy = {original_list is deep_copied_list}")

# 修改深拷贝后的列表
deep_copied_list[0] = 100  # 不会影响原列表
print(f"修改deep_copied_list[0] = 100")
print(f"修改deep_copied_list后,original_list={original_list}")      # [1, 2, [3, 4]]
print(f"修改deep_copied_list后,deep_copied_list={deep_copied_list}")   # [100, 2, [3, 4]]

# 修改深拷贝后的原列表
original_list[2][0] = 300  # 也不会影响原列表中的子列表
print(f"修改original_list[2][0] = 300 ")
print(f"修改original_list后,修改original_list={original_list}")      # [1, 2, [300, 4]]
print(f"修改original_list后,deep_copied_list={deep_copied_list}")   # [100, 2, [3, 4]]
print("深copy的特点是:修改各自的值，各自不受影响")
print("深拷贝会递归复制对象及其所有子对象，创建一个完全独立的新对象")
