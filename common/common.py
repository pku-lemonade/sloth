import simpy

from common.runtime_config import MonitoringConfig

class MonitoredResource(simpy.Resource):
    def __init__(self, env, capacity=1, monitoring_config: MonitoringConfig | None = None):
        super().__init__(env, capacity=capacity)
        self.data = []
        self.monitoring_config = monitoring_config or MonitoringConfig()

    def checkneed(self):
        return self.monitoring_config.should_record(self._env.now)

    def exe(self, task, delay, ins, v=None, core=None, index=None, opcode=None, flops=None, attributes=None):
        req = super().request()
        yield req
        if v is not None:
            v.run(core, index, ins.layer_id, opcode, flops=flops)
        if self.checkneed():
            if attributes is None:
                self.data.append((task, self._env.now, len(self.queue), "req", "B"))
            else:
                self.data.append((task, self._env.now, len(self.queue), "req", "B", attributes))
        if v is None:
            yield self._env.timeout(delay)
        else:
            yield self._env.timeout(delay, value=v)
        if self.checkneed():
            if attributes is None:
                self.data.append((task, self._env.now, len(self.queue), "req", "E"))
            else:
                self.data.append((task, self._env.now, len(self.queue), "req", "E", attributes))
                
        super().release(req)

    def execute(self, task, delay, ins, v=None, core=None, index=None, opcode=None, flops=None, attributes=None):
        return self._env.process(self.exe(task, delay, ins, v=v, core=core, index=index, opcode=opcode, flops=flops, attributes=attributes))
