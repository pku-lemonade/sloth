import json
import os
import re
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from typing import List, Dict, Tuple
from pydantic import BaseModel
from tracer.root_cause_analysis import FailSlow, FailSlows
from compiler.instruction_generator import config_analyzer
from common.arch_config import ArchConfig, NoCConfig
from evaluater.sim_type import Direction

def match_failures(detected: Dict, truth: Dict) -> bool:
    intersection_start = max(detected["start_time"], truth["start_time"])
    intersection_end = min(detected["end_time"], truth["end_time"])
    intersection = max(0, intersection_end - intersection_start)
    union = (detected["end_time"] - detected["start_time"]) + (truth["end_time"] - truth["start_time"]) - intersection
    IoU = intersection / union if union > 0 else 0.0

    if detected["kind"] == "pe":
        return detected["id"] == truth["pe_id"]
    elif detected["kind"] == "link":
        return detected["id"] == truth["router_id"] and detected["dst_id"] == truth["dst_id"]


def evaluate_detection(ground_truth: List[Dict], detected: List[Dict], config: NoCConfig) -> Dict[str, float]:
    TN, TP, FP, FN = 0, 0, 0, 0
    failure_sum = 0
    match_num = 0

    for i in range(len(ground_truth)):
        truth = ground_truth[i]
        detected_item = detected[i]['data']

        tmp = 0
        for failure_type in truth:
            for failure in truth[failure_type]:
                tmp = tmp + 1
        failure_sum += tmp
        
        for item in detected_item:
            TP_added = False
            if item['kind'] == 'pe':
                for failure in truth['lsu']:
                    matched = match_failures(item, failure)
                    if matched:
                        match_num += 1
                        if not TP_added:
                            TP += 1
                            TP_added = True

                for failure in truth['tpu']:
                    matched = match_failures(item, failure)
                    if matched:
                        match_num += 1
                        if not TP_added:
                            TP += 1
                            TP_added = True
            
            if item['kind'] == 'link':
                for failure in truth['link']:
                    matched = match_failures(item, failure)
                    if matched:
                        match_num += 1
                        if not TP_added:
                            TP += 1
                            TP_added = True

            if not TP_added:
                FP += 1

    TN = failure_sum
    FN = failure_sum - match_num

    total = TP + TN + FP + FN

    return {
        "TN": TN,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "accuracy": (TP + TN) / total if total > 0 else 0,
        "FPR": FP / (FP + TN) if (FP + TN) > 0 else 0,
        "FNR": FN / (TP + FN) if (TP + FN) > 0 else 0,
    }

def load_multiple_json(folder_path: str) -> List[Dict]:
    all_data = []
    for root, _, files in os.walk(folder_path):
        files.sort(key = lambda f: int(re.search(r'(\d+)\.json$', f).group(1)))
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)  
                    all_data.append(data)
    return all_data

def cal_dst(router_id: int, direction: int, config: NoCConfig) -> int:
    match direction:
        case Direction.NORTH:
            return router_id + 1
        case Direction.SOUTH:
            return router_id - 1
        case Direction.EAST:
            return router_id + config.y
        case Direction.WEST:
            return router_id - config.y
        case _:
            return -1  

parser = argparse.ArgumentParser()
if __name__ == "__main__":
    parser.add_argument("--arch", type=str, default="arch/gemini4_4.json", help="Path to the architecture configuration file")
    parser.add_argument("--output", type=str, default=None, help="Path to write the result")
    parser.add_argument("ground_truth", type=str, help="Path to the ground truth dataset")
    parser.add_argument("detected", type=str, help="Path to the detected failures dataset")

    args = parser.parse_args()

    arch_configs = config_analyzer(args.arch)
    ground_truth = load_multiple_json(args.ground_truth)
    detected = load_multiple_json(args.detected)

    for case in ground_truth:
        for failure in case['link']:
            failure["dst_id"] = cal_dst(failure["router_id"], failure["direction"], arch_configs.noc)
            
            if failure["router_id"] > failure["dst_id"]:
                failure["router_id"], failure["dst_id"] = failure["dst_id"], failure["router_id"]

    metrics = evaluate_detection(ground_truth, detected, arch_configs.noc)
    if args.output is None:
        print(f"result: {metrics}")
    else:
        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        merged_data = {**existing_data, **metrics}

        with open(args.output, 'w') as f:
            json.dump(merged_data, f, indent=4)

