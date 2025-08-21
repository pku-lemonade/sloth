import os
import sys
import copy
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from evaluater.sim_type import Task, Probe

compute_code = ["Conv", "Pool", "FC", "Elem", "GConv", "PTP", "Trans"]
communication_code = ["Send", "Recv"]
io_code = ["Read", "Write"]

def fail_probing(
    tasks: list[Task],
    fragment: str,
    type: str,
    location: str,
    level: str,
    structure: str,
) -> list:
    assert fragment in ("Exec", "Route", "Mem"), f"Unsupported fragment: {fragment}"
    assert type in ("Comm", "Comp", "IO"), f"Unsupported type: {type}"
    assert location in ("Post", "Pre", "Surround"), f"Unsupported location: {location}"
    assert level in ("Inst", "Stage"), f"Unsupported level: {level}"
    assert structure in ("List", "Sketch"), f"Unsupported structure: {structure}"

    PreExec = { "start_time": -1, "flops": -1 }
    PostExec = { "end_time": -1 }
    PreRoute = { "start_time": -1, "data_size": -1, "src_id": -1 }
    PostRoute = { "end_time": -1, "dst_id": -1 }
    PreMem = { "start_time": -1, "data_size": -1 }
    PostMem = { "end_time": -1 }

    for inst in tasks:
        inst_type = inst.opcode
        layer_id = inst.layer_id
        index = inst.index

        if location in ("Pre", "Surround"):
            if inst_type in compute_code and type == "Comp":
                match fragment:
                    case "Exec":
                        inst.probe_st = Probe(flag = 0, metric = PreExec)
                    case "Route":
                        inst.probe_st = Probe(flag = 0, metric = PreRoute)
                    case "Mem":
                        inst.probe_st = Probe(flag = 0, metric = PreMem)
            elif inst_type in communication_code and type == "Comm":
                match fragment:
                    case "Exec":
                        inst.probe_st = Probe(flag = 0, metric = PreExec)
                    case "Route":
                        inst.probe_st = Probe(flag = 0, metric = PreRoute)
                    case "Mem":
                        inst.probe_st = Probe(flag = 0, metric = PreMem)
            elif inst_type in io_code and type == "IO":
                match fragment:
                    case "Exec":
                        inst.probe_st = Probe(flag = 0, metric = PreExec)
                    case "Route":
                        inst.probe_st = Probe(flag = 0, metric = PreRoute)
                    case "Mem":
                        inst.probe_st = Probe(flag = 0, metric = PreMem)


        if location in ("Post", "Surround"):
            if inst_type in compute_code and type == "Comp":
                match fragment:
                    case "Exec":
                        inst.probe_ed = Probe(flag = 1, metric = PostExec)
                    case "Route":
                        inst.probe_ed = Probe(flag = 1, metric = PostRoute)
                    case "Mem":
                        inst.probe_ed = Probe(flag = 1, metric = PostMem)
            elif inst_type in communication_code and type == "Comm":
                match fragment:
                    case "Exec":
                        inst.probe_ed = Probe(flag = 1, metric = PostExec)
                    case "Route":
                        inst.probe_ed = Probe(flag = 1, metric = PostRoute)
                    case "Mem":
                        inst.probe_ed = Probe(flag = 1, metric = PostMem)
            elif inst_type in io_code and type == "IO":
                match fragment:
                    case "Exec":
                        inst.probe_ed = Probe(flag = 1, metric = PostExec)
                    case "Route":
                        inst.probe_ed = Probe(flag = 1, metric = PostRoute)
                    case "Mem":
                        inst.probe_ed = Probe(flag = 1, metric = PostMem)

    return tasks
