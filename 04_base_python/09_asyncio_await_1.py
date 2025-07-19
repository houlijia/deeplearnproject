# ---- 协程用法 ----
print("---- 1、协程基本用法 ----")
import asyncio
async def say_hello():
    print("Hello")
    await asyncio.sleep(1)
    print("world")
# 运行协程
asyncio.run(say_hello())


# ---- 多个协程并发执行 ----
print("---- 2、多个协程并发执行 ----")
async def task(name, seconds):
    print(f"Task {name} started")
    await asyncio.sleep(seconds)
    print(f"Task {name} completed")

async def runtask():
    await asyncio.gather(
        task("C", 1),
        task("B", 3),
        task("A", 2)
    )

asyncio.run(runtask())

# 协程与任务(Task)
print("---- 3、协程与任务(Task) ----")
async def my_coroutine():
    return 110

async def runtask():
    # 将协程包装成任务
    task1 = asyncio.create_task(my_coroutine())
    # 等待任务完成
    result = await task1
    print(result)

asyncio.run(runtask())

