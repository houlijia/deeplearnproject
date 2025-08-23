import random


def random_access(nums:list[int]) -> int:
    random_index = random.randint(0, len(nums)-1)
    random_num = nums[random_index]
    return random_num





"""Driver Code"""
if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5, 6]
    a01 = nums.index(2)
    print(f"a01 = {a01}")
    aa: list
    result = random_access(nums)
    print(f"result = {result}")
