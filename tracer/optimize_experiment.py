import os
import json
import math
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    
    parser.add_argument("-a", "--alpha", type=float, required=True, help="Alpha value")
    parser.add_argument("-b", "--beta", type=float, required=True, help="Beta value")
    parser.add_argument("-g", "--gamma", type=float, required=True, help="Gamma value")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Input directory")

    args = parser.parse_args()

    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    input_dir = args.input_dir

    output_file = os.path.join(input_dir, "results.txt")
    print(f"Output will be saved to: {output_file}")
    lowest_cost = float("inf")
    best_case_path = ""
    best_case_name = ""

    with open(output_file, "w") as output:
        output.write("Case\tACC\tDM\tR\tCOST\n")

        for root, dirs, files in os.walk(input_dir):
            if "overall.json" in files:
                overall_json_path = os.path.join(root, "overall.json")

                with open(overall_json_path, "r") as json_file:
                    data = json.load(json_file)

                ACC = data.get("accuracy", 0)
                DM = data.get("overhead", 0)
                R = data.get("rate", 0)

                if ACC > 0 and DM > 0 and R > 0:
                    COST = (ACC ** alpha) * (DM ** beta) * (R ** gamma)
                else:
                    COST = float("inf")  

                case_name = os.path.relpath(root, input_dir)

                output.write(f"{case_name}\t{ACC}\t{DM}\t{R}\t{COST}\n")

                if COST < lowest_cost:
                    lowest_cost = COST
                    best_case_path = root
                    best_case_name = case_name
                
        output.write(f"\nLowest cost case: {best_case_name}\nscore: {lowest_cost}\n")
if __name__ == "__main__":
    main()