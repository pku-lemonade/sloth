import simpy
import os

class MySimulator:
    def __init__(self, env):
        self.env = env

    def sample_process(self, name, duration):
        print(f"{self.env.now}: Process {name} started.")
        # 模拟处理任务：等待 duration 时间
        yield self.env.timeout(duration)
        print(f"{self.env.now}: Process {name} completed.")

    def add(self):
        print(1+1)
        yield self.env.timeout(1)

    def run_process(self, name, duration):
        proc = self.env.process(self.add())
        return proc

# 使用示例
if __name__ == "__main__":
    a=[1,2,3]
    b=a+[4]
    print(b)
    os.mkdir("gen")