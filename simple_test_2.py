import sys

# 强制刷新输出
sys.stdout.write("HELLO\n")
sys.stdout.flush()

with open("test_simple.txt", "w") as f:
    f.write("Hello from Python\n")
    f.flush()
