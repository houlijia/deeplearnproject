import asyncio
from time import perf_counter
import aiohttp  # 网络的异步
import aiofiles


async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()


async def read_file(filepath):
    async with aiofiles.open(filepath, 'r') as f:
        return await f.read()


# 定义协程1
async def fun01():
    print("这是 fun01 函数的准备动作")
    await asyncio.sleep(1)
    print("这是 fun01 函数的ok动作")
    return "fun01"


# 定义协程2
async def fun02():
    print("这是 fun02 函数的准备动作")
    print(f"await执行结果: {await asyncio.sleep(1)}")
    print("这是 fun02 函数的ok动作")
    return "fun02"


# 此处main也是一个协程函数
async def main():
    # 实例 1:
    # url = 'https://jsonplaceholder.typicode.com/posts'
    # filepath = './README.txt'
    # result00 = asyncio.as_completed([fetch_url(url), read_file(filepath)])
    # for i in result00:
    #     print(await i)


    # 实例 2:
    # 将协程包装成任务,方式一：手动方式
    # task1 = asyncio.create_task(fun01())
    # task2 = asyncio.create_task(fun02())
    # # 使用await获取结果
    # result1 = await task1
    # result2 = await task2
    # print(result1)
    # print(result2)

    # 方式二：自动方式
    result_total = await asyncio.gather(fun01(), fun02())
    for i in result_total:
        print(i)
    # gather 是等所有的协程执行完毕之后一起返回

    # 方式三：自动方式
    # result_complted 是一个迭代器
    # result_complted = asyncio.as_completed([fun01(), fun02()])  # 参数是一个数组
    # for i in result_complted:
    #     print(await i)
    # as_completed 的特点是不会等到所有的协程都完成，一有协程运行完，就返回一个结果

start_time = perf_counter()
# 建立事件循环 1、将协程作为输入，然后创建一个事件循环，将接收到的协程放入事件循环中
asyncio.run(main())
end_time = perf_counter()
print(f"Time:{end_time - start_time:.2f} s")
