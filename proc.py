import sys
import os
import glob
import json

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} {{lsu|tpu|spm|flow}} {{index|all}}")
        sys.exit(1)

    type_ = sys.argv[1]
    param = sys.argv[2]

    os.makedirs('build', exist_ok=True)

    if param != "all":
        src_file = f"gen/{type_}{param}.json"
        dst_file = f"build/{type_}{param}.json"

        if not os.path.isfile(src_file):
            print(f"File {src_file} does not exist")
            sys.exit(1)

        with open(src_file, 'r') as f:
            content = f.read()

        with open(dst_file, 'w') as f:
            f.write('[\n')
            f.write(content)
            f.write('\n]\n')

        print(f"File {dst_file} wrapped with [ and ]")

    else:
        if type_ != "flow":
            files = sorted(glob.glob(f"gen/{type_}*.json"))
        else:
            files = sorted(glob.glob(f"gen/{type_}_out*.json")) + sorted(glob.glob(f"gen/{type_}_in*.json"))

        if not files:
            print(f"No files found to merge for type {type_}")
            sys.exit(1)

        merged_file = f"build/{type_}_merged.json"
        json_objects = []

        for file in files:
            with open(file, 'r') as f:
                try:
                    data = json.load(f)
                    json_objects.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: {file} is not valid JSON. Treating as raw text.")
                    f.seek(0)
                    raw_content = f.read()
                    json_objects.append(json.loads(f"[{raw_content}]")[0])

        with open(merged_file, 'w') as f:
            f.write('[\n')
            for idx, obj in enumerate(json_objects):
                for idy, item in enumerate(obj):
                    json_str = json.dumps(item, separators=(',', ':'))
                    f.write(json_str)
                    if idx != len(json_objects) - 1 or idy != len(obj) - 1:
                        f.write(',\n')
                    else:
                        f.write('\n')
            f.write(']\n')

        print(f"Merged file created: {merged_file}")

if __name__ == "__main__":
    main()