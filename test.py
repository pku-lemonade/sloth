import simpy

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
    env = simpy.Environment()
    sim = MySimulator(env)
    
    # 启动两个 process
    p1 = env.process(sim.run_process("A", 5))
    
    # 运行仿真，直到所有 process 执行完毕
    env.run()