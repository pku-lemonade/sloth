import simpy

def proc(env):
    print(f'proc started at {env.now}')
    yield env.timeout(3)
    print(f'proc resumed at {env.now}')

def main(env):
    print(f'main started at {env.now}')
    env.process(proc(env))  # 启动 proc 但不 yield
    print(f'main continues at {env.now}')
    yield env.timeout(5)
    print(f'main finished at {env.now}')

env = simpy.Environment()
env.process(main(env))
env.run()
