#!/bin/bash
# ./proc.sh lsu all means print all lsu
# ./proc.sh lsu 0 means print lsu0.json

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 {lsu|tpu|spm} {index|all}"
  exit 1
fi
type="$1"
param="$2"

# 创建build目录
mkdir -p build

# 如果第二个参数不是 "all"，则只处理单个文件
if [ "$param" != "all" ]; then
  src_file="gen/${type}${param}.json"
  dst_file="build/${type}${param}.json"

  if [ ! -f "$src_file" ]; then
    echo "File $src_file does not exist"
    exit 1
  fi

  content=$(cat "$src_file")
  # 在开头加 [ ，在结尾加 ] ，输出到 build 目录中的新文件
  printf "[\n%s\n]\n" "$content" > "$dst_file"
  echo "File $dst_file wrapped with [ and ]"

else
  # 如果第二个参数为 "all"，则合并 gen/${type}*.json 中的所有文件
  merged_file="build/${type}_merged.json"
  first=1
  output=""
  if [ "${type}" != "flow" ]; then
  for f in gen/"${type}"*.json; do
    if [ "$first" -eq 1 ]; then
      output+="$(cat "$f")"
      first=0
    else
        output+=","
        output+=$'\n'
        output+="$(cat "$f")"
    fi
  done
    else
        for f in gen/"${type}_out"*.json; do
    if [ "$first" -eq 1 ]; then
      output+="$(cat "$f")"
      first=0
    else
        output+=","
        output+=$'\n'
        output+="$(cat "$f")"
    fi
  done
  for f in gen/"${type}_in"*.json; do
    if [ "$first" -eq 1 ]; then
      output+="$(cat "$f")"
      first=0
    else
        output+=","
        output+=$'\n'
        output+="$(cat "$f")"
    fi
  done
    fi

  printf "[\n%s\n]\n" "$output" > "$merged_file"
  echo "Merged file created: $merged_file"
fi

