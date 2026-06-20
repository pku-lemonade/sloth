import numpy as np


def calc_window(windows):
    stats = []
    for start_time, event in windows:
        if not event:
            continue
        avg = sum(event) / len(event)
        std = (sum((exe_time - avg) ** 2 for exe_time in event) / len(event)) ** 0.5
        stats.append((start_time, avg, std))
    return stats


def interval_merge(failslow):
    interval_begin = 0
    merged_failslow = []

    for idx in range(1, len(failslow)):
        if failslow[idx - 1][1] + 1 < failslow[idx][0]:
            merged_failslow.append((failslow[interval_begin][0], failslow[idx - 1][1]))
            interval_begin = idx
    merged_failslow.append((failslow[interval_begin][0], failslow[-1][1]))

    return merged_failslow


def softmax(x, beta=1.0):
    x = np.array(x)
    e_x = np.exp(beta * (x - np.max(x)))
    return e_x / e_x.sum()
