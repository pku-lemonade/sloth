import os


for i in range(1, 3):
    simulation_cmd = f"python d:/University/PKU/manycore-simpy/run.py --workload tests/darknet19/workload.json --arch arch/gemini4_4.json --times 16 --fail failslow/dataset/fail{i}.json"
    os.system(simulation_cmd)
    analysis_cmd = f"python d:/University/PKU/manycore-simpy/analysis/comm_fail.py --network tests/darknet19/mapping.json --arch arch/gemini4_4.json --report data/darknet19/experiment/fail{i}/report.json data/darknet19/experiment/normal data/darknet19/experiment/fail{i}"
    os.system(analysis_cmd)
