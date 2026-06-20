import argparse
import json
import os
import re
import sys
import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from typing import Dict, List, Tuple

from compiler.instruction_generator import config_analyzer
from tracer.topology import create_topology

FAIL_JSON_RE = re.compile(r"fail(\d+)\.json$")


def match_failures(detected: Dict, truth: Dict) -> bool:
    intersection_start = max(detected["start_time"], truth["start_time"])
    intersection_end = min(detected["end_time"], truth["end_time"])
    intersection = max(0, intersection_end - intersection_start)
    union = (detected["end_time"] - detected["start_time"]) + (truth["end_time"] - truth["start_time"]) - intersection
    iou = intersection / union if union > 0 else 0.0

    if detected["kind"] == "pe":
        return detected["id"] == truth["pe_id"]
    if detected["kind"] == "link":
        return detected["id"] == truth["router_id"] and detected["dst_id"] == truth["dst_id"]
    return False


def evaluate_detection(ground_truth: List[Dict], detected: List[Dict], config=None) -> Dict[str, float]:
    tn, tp, fp, fn = 0, 0, 0, 0
    failure_sum = 0
    match_num = 0

    for index in range(len(ground_truth)):
        truth = ground_truth[index]
        detected_item = detected[index]["data"]

        tmp = 0
        for failure_type in truth:
            for _failure in truth[failure_type]:
                tmp += 1
        failure_sum += tmp

        for item in detected_item:
            tp_added = False
            if item["kind"] == "pe":
                for failure in truth["lsu"]:
                    matched = match_failures(item, failure)
                    if matched:
                        match_num += 1
                        if not tp_added:
                            tp += 1
                            tp_added = True

                for failure in truth["tpu"]:
                    matched = match_failures(item, failure)
                    if matched:
                        match_num += 1
                        if not tp_added:
                            tp += 1
                            tp_added = True

            if item["kind"] == "link":
                for failure in truth["link"]:
                    matched = match_failures(item, failure)
                    if matched:
                        match_num += 1
                        if not tp_added:
                            tp += 1
                            tp_added = True

            if not tp_added:
                fp += 1

    tn = failure_sum
    fn = failure_sum - match_num
    total = tp + tn + fp + fn

    return {
        "TN": tn,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "accuracy": (tp + tn) / total if total > 0 else 0,
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "FNR": fn / (tp + fn) if (tp + fn) > 0 else 0,
    }


def fail_name_key(name: str) -> Tuple[int, str]:
    match = FAIL_JSON_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported fail-case filename: {name}")
    return int(match.group(1)), name


def load_failure_json_map(folder_path: str) -> Dict[str, Dict]:
    all_data = {}
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if FAIL_JSON_RE.fullmatch(file_name) is None:
                continue
            with open(os.path.join(root, file_name), "r", encoding="utf-8") as file:
                all_data[file_name] = json.load(file)
    return all_data


def normalize_ground_truth_links(ground_truth: List[Dict], topology) -> None:
    for case in ground_truth:
        valid_links = []
        for failure in case.get("link", []):
            try:
                failure["dst_id"] = topology.neighbor_for_direction(failure["router_id"], failure["direction"])
            except ValueError as exc:
                warnings.warn(
                    f"Skipping invalid ground-truth link failure router_id={failure['router_id']} "
                    f"direction={failure['direction']}: {exc}"
                )
                continue
            if failure["router_id"] > failure["dst_id"]:
                failure["router_id"], failure["dst_id"] = failure["dst_id"], failure["router_id"]
            valid_links.append(failure)
        case["link"] = valid_links


def evaluate_detection_folders(
    ground_truth_folder: str,
    detected_folder: str,
    arch_path: str,
    case_names: List[str] | None = None,
) -> Dict[str, float]:
    arch_configs = config_analyzer(arch_path)
    topology = create_topology(arch_configs)

    ground_truth_map = load_failure_json_map(ground_truth_folder)
    detected_map = load_failure_json_map(detected_folder)

    if case_names is None:
        selected_case_names = sorted(ground_truth_map, key=fail_name_key)
    else:
        selected_case_names = sorted(
            [f"{name}.json" if FAIL_JSON_RE.fullmatch(name) is None else name for name in case_names],
            key=fail_name_key,
        )

    missing_ground_truth = sorted([name for name in selected_case_names if name not in ground_truth_map], key=fail_name_key)
    missing_detected = sorted([name for name in selected_case_names if name not in detected_map], key=fail_name_key)
    unexpected_detected = sorted(set(detected_map) - set(selected_case_names), key=fail_name_key)
    if missing_ground_truth or missing_detected or unexpected_detected:
        details = []
        if missing_ground_truth:
            details.append(f"missing ground-truth cases: {', '.join(missing_ground_truth)}")
        if missing_detected:
            details.append(f"missing detected reports: {', '.join(missing_detected)}")
        if unexpected_detected:
            details.append(f"unexpected detected reports: {', '.join(unexpected_detected)}")
        raise ValueError("; ".join(details))

    ground_truth = [ground_truth_map[name] for name in selected_case_names]
    detected = [detected_map[name] for name in selected_case_names]
    normalize_ground_truth_links(ground_truth, topology)
    return evaluate_detection(ground_truth, detected, arch_configs.noc)


parser = argparse.ArgumentParser()
if __name__ == "__main__":
    parser.add_argument("--arch", type=str, default="arch/gemini4_4.json", help="Path to the architecture configuration file")
    parser.add_argument("--output", type=str, default=None, help="Path to write the result")
    parser.add_argument("ground_truth", type=str, help="Path to the ground truth dataset")
    parser.add_argument("detected", type=str, help="Path to the detected failures dataset")

    args = parser.parse_args()
    metrics = evaluate_detection_folders(args.ground_truth, args.detected, args.arch)
    if args.output is None:
        print(f"result: {metrics}")
    else:
        if os.path.exists(args.output):
            with open(args.output, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        else:
            existing_data = {}

        merged_data = {**existing_data, **metrics}
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(merged_data, file, indent=4)
