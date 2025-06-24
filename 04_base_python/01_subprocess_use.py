import subprocess

# 1.向子进程发送输入
p = subprocess.Popen(['bc', '-l'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
p.stdin.write("2+2\n")
p.stdin.flush()

for _ in range(1):
    print(p.stdout.readline().strip())

print(p.stdin.close())
print(p.wait())

# 2.超时控制
try:
    p = subprocess.Popen(["sleep", "10"])
    p.wait(timeout=3)
except subprocess.TimeoutExpired as e:
    print(f"error: {e}")  # Command '['sleep', '10']' timed out after 3 seconds
    p.kill()
    p.wait(timeout=1)
    print("进程已经kill")

# 3.进程间管道通信
# 生成数据
p1 = subprocess.Popen(["python", "-c", "for i in range(5): print(i)"], stdout=subprocess.PIPE)

# 处理数据
p2 = subprocess.Popen(['python', '-c', "import sys; print('平方：',[int(line)**2 for line in sys.stdin])"],
                      stdin=p1.stdout,
                      stdout=subprocess.PIPE)
p1.stdout.close()



