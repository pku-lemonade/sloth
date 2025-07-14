from typing import List, Dict, Tuple
from pydantic import BaseModel
from analysis.comm_fail import FailSlow, FailSlows
from tools.geninst_new import config_analyzer
from src.arch_config import ArchConfig, NoCConfig
from src.sim_type import Direction
import json
import os
import argparse

THRESHOLD = 0.7
def match_failures(detected: Dict, truth: Dict) -> bool:
    intersection_start = max(detected["start_time"], truth["start_time"])
    intersection_end = min(detected["end_time"], truth["end_time"])
    intersection = max(0, intersection_end - intersection_start)
    union = (detected["end_time"] - detected["start_time"]) + (truth["end_time"] - truth["start_time"]) - intersection
    IoU = intersection / union if union > 0 else 0.0
    if IoU < THRESHOLD:
        return False
    
    if detected["kind"] == "pe":
        return detected["id"] == truth["id"]
    elif detected["kind"] == "link":
        return detected["id"] == truth["router_id"] and detected["dst_id"] == truth["dst_id"]


def evaluate_detection(ground_truth: List[Dict], detected: List[Dict], config: NoCConfig) -> Dict[str, float]:
    #ground_truth=[{'router': [], "link": [{'start_time':,...,...}], }, ...]
    #list of dicts, each dict contains 'router' and 'link' keys,each value is a list of dicts
    #detected=[{'data': [{'kind':,'id':,...,}{}] }, ...]
    #list of dicts, each dict contains 'data' key, value is a list of dicts with 'kind', 'id', etc.
    TN, TP, FP, FN = 0, 0, 0, 0
    truth_matched = [False] * len(ground_truth)
    failure_sum=0
    for i in range(len(ground_truth)):
        truth = ground_truth[i]# Dict with 'router' and 'link' keys, each value is a list of dicts
        detected_item = detected[i]['data']#List of dicts with 'kind', 'id', etc.
        tmp=0
        for failure_type in truth:
            for failure in truth[failure_type]:
                tmp=tmp+1
        failure_sum += tmp
        if tmp == 0:
            TN += 1
        for item in detected_item:
            matched = False
            if item['kind'] == 'pe':
                for failure in truth['lsu']:
                    matched = match_failures(item, failure)
                    if matched:
                        TP += 1
                        break
                if not matched:
                    for failure in truth['tpu']:
                        matched = match_failures(item, failure)
                        if matched:
                            TP += 1
                            break
            if item['kind'] == 'link':
                for failure in truth['link']:
                    matched = match_failures(item, failure)
                    if matched:
                        TP += 1
                        break
            if not matched:
                FP += 1
                    
    
    total = failure_sum + FP + TN
    FN = failure_sum - TP  
    return {
        "accuracy": (TP + TN) / total if total > 0 else 0,
        "FPR": FP / (total-failure_sum) if (total-failure_sum) > 0 else 0,
        "FNR": FN / failure_sum if failure_sum > 0 else 0,
    }

def load_multiple_json(folder_path: str) -> List[Dict]:
    all_data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)  
                    all_data.append(data)
    return all_data

def cal_dst(router_id: int, direction: int, config: NoCConfig) -> int:
    match direction:
        case Direction.NORTH:
            return router_id - config.y
        case Direction.SOUTH:
            return router_id + config.y
        case Direction.EAST:
            return router_id + 1
        case Direction.WEST:
            return router_id - 1
        case _:
            return -1  
    
parser = argparse.ArgumentParser()
if __name__ == "__main__":
    parser.add_argument("--arch", type=str, default="arch/gemini4_4.json", help="Path to the architecture configuration file")
    parser.add_argument("ground_truth", type=str, help="Path to the ground truth dataset")
    parser.add_argument("detected", type=str, help="Path to the detected failures dataset")
    args = parser.parse_args()
    arch_configs = config_analyzer(args.arch)
    ground_truth = load_multiple_json(args.ground_truth)
    detected = load_multiple_json(args.detected)
    
    for case in ground_truth:
        for failure in case['link']:
            failure["dst_id"] = cal_dst(failure["router_id"],failure["direction"],arch_configs.noc)
    
    metrics = evaluate_detection(ground_truth, detected, arch_configs.noc)
    print(f"result: {metrics}")