import simpy

class MonitoredResource(simpy.Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def request(self, *args, **kwargs):
        req=super().request(*args, **kwargs)
        self.data.append((self._env.now, len(self.queue)))
        return req
        

    def release(self, *args, **kwargs):
        self.data.append((self._env.now, len(self.queue)))
        super().release(*args, **kwargs)

def test_process(env, res):
    req=res.request()
    yield req
    yield env.timeout(1)
    res.release(req)

env = simpy.Environment()

res = MonitoredResource(env, capacity=1)
p1 = env.process(test_process(env, res))
p2 = env.process(test_process(env, res))
env.run()

print(res.data)