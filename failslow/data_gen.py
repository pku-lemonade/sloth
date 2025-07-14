import random
import os
import sys
import json
from typing import List
from enum import IntEnum
from pydantic import BaseModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.sim_type import Direction, RouterFail, LinkFail, LsuFail, TpuFail, FailSlow

def check(pe_id, Direction):
    if Direction == Direction.NORTH:
        return pe_id % 4 != 3
    elif Direction == Direction.SOUTH:
        return pe_id % 4 != 0
    elif Direction == Direction.EAST:
        return pe_id // 4 != 3
    else:
        return pe_id // 4 != 0

# 生成单个数据点
def generate_single_fail_slow(fail_type: str, time_range: tuple[int, int], id_range: tuple[int, int]) -> FailSlow:
    start_time = random.randint(*time_range)
    # 持续时间为 0.1s 到 5s
    duration = random.randint(10000000, 500000000)
    end_time = start_time + duration
    # 失速倍率
    times = random.randint(5, 50)

    router_id = random.randint(*id_range)
    pe_id = random.randint(*id_range)

    fail_data = {
        'router': [],
        'link': [],
        'lsu': [],
        'tpu': [],
    }

    if fail_type == 'router':
        fail_data['router'].append(
            RouterFail(
                start_time = start_time,
                end_time = end_time,
                router_id = router_id,
                times = times
            )
        )
    elif fail_type == 'link':
        direction = random.choice(list(Direction))

        # 确保链路一定存在
        while not check(pe_id, direction):
            direction = random.choice(list(Direction))

        fail_data['link'].append(
            LinkFail(
                start_time = start_time,
                end_time = end_time,
                router_id = router_id,
                direction = direction,
                times = times
            )
        )
    elif fail_type == 'lsu':
        fail_data['lsu'].append(
            LsuFail(
                start_time = start_time,
                end_time = end_time,
                pe_id = pe_id,
                times = times
            )
        )
    elif fail_type == 'tpu':
        fail_data['tpu'].append(
            TpuFail(
                start_time = start_time,
                end_time = end_time,
                pe_id = pe_id,
                times = times
            )
        )
    else:
        raise ValueError("Invalid fail_type. Choose from 'router', 'link', 'lsu', 'tpu'.")

    return FailSlow(**fail_data)


# 批量生成数据
def generate_dataset(num_samples: int, fail_ratios: dict, output_dir: str = "failslow/dataset"):
    os.makedirs(output_dir, exist_ok=True)

    # 计算各种 fail 数量
    total_ratio = sum(fail_ratios.values())
    fail_counts = {k: int(num_samples * (v / total_ratio)) for k, v in fail_ratios.items()}

    # 计算剩余数据点
    assigned = sum(fail_counts.values())
    remaining = num_samples - assigned
    if remaining > 0:
        # 补齐数据点
        residuals = {k: num_samples * (v / total_ratio) - fail_counts[k] for k, v in fail_ratios.items()}
        sorted_keys = sorted(residuals.keys(), key=lambda k: residuals[k], reverse=True)
        for i in range(remaining):
            fail_counts[sorted_keys[i % len(sorted_keys)]] += 1

    print(f"数据集包含：{fail_counts}")

    # 失速起始时间 0s 到 2s
    time_range = (0, 200000000)
    id_range = (0, 16)
    # 生成数据点
    index = 1
    for fail_type, count in fail_counts.items():
        for _ in range(count):
            sample = generate_single_fail_slow(fail_type, time_range, id_range)
            file_path = os.path.join(output_dir, f"fail{index}.json")
            with open(file_path, 'w') as f:
                f.write(sample.model_dump_json(indent=4))
            index += 1


# ==== 执行 ====
if __name__ == "__main__":
    # 设置总样本数和比例（可修改）
    total_samples = 119
    fail_ratios = {
        'router': 0,
        'link': 0.7,
        'lsu': 0,
        'tpu': 0.3
    }
    generate_dataset(total_samples, fail_ratios)
