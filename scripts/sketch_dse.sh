#!/bin/bash

evaluate(){
    hash=$1
    bucket=$2
    size=$3
    threshold=$4

    case_path=$5

    i=1
    for f in $(ls failslow/dataset/*.json | sort -V); do
        echo "$i $hash $bucket $size $threshold $case_path"
        ((i++))
    done > file_list.txt

    log_path=$case_path/logs
    mkdir -p $log_path

    cat file_list.txt | parallel -j 64 --colsep ' ' "./single_case.sh {1} {2} {3} {4} {5} {6} > $log_path/fail{1}.log 2>&1"

    python analysis/compare.py --output "$case_path/overall.json" "data/dataset" "$case_path/report"
}

dse(){
    echo "************************* Start DSE *************************"
    hash_range=(1 3 5 7 9)
    bucket_range=(1024 2048 4096 8192 16384)
    size_range=(16384 20480 24576 28672 32768)
    threshold_range=(5 10 15 20 25)

    st_time=$(date +%s)

    for num_hash in ${hash_range[*]}; do
        temp_path0=$output_path/hash_$num_hash
        mkdir -p $temp_path0

        for num_bucket in ${bucket_range[*]}; do
            temp_path1=$temp_path0/bucket_$num_bucket
            mkdir -p $temp_path1

            for stage_size in ${size_range[*]}; do
                temp_path2=$temp_path1/size_$stage_size
                mkdir -p $temp_path2

                for threshold in ${threshold_range[*]}; do
                    temp_path3=$temp_path2/threshold_$threshold
                    mkdir -p $temp_path3

                    echo "========================= Evaluate Start ========================="

                    start_time=$(date +%s)
                    start_date=$(date "+%F %T")
                    echo "$start_date: Start evaluating case[$num_hash-$num_bucket-$stage_size-$threshold]"

                    evaluate $num_hash $num_bucket $stage_size $threshold $temp_path3

                    end_time=$(date +%s)
                    end_date=$(date "+%F %T")
                    echo "$end_date: Finish evaluating case[$num_hash-$num_bucket-$stage_size-$threshold]"

                    eval_time=$((end_time - start_time))
                    _second=`expr $eval_time % 60`
                    _minute=`expr $eval_time / 60`
                    _hour=`expr $_minute / 60`
                    _minute=`expr $_minute % 60`
                    echo "Evaluating time: $_hour h $_minute m $_second s ($eval_time)seconds."
                    echo "========================= Evaluate End ========================="
                done
            done
        done
    done

    ed_time=$(date +%s)

    dse_time=$((ed_time - st_time))
    second=`expr $dse_time % 60`
    minute=`expr $dse_time / 60`
    hour=`expr $minute / 60`
    minute=`expr $minute % 60`
    echo "DSE time: $hour h $minute m $second s ($dse_time)seconds."

    echo "************************* Finish DSE *************************"
}

output_path=""
mkdir -p $output_path

dse > $output_path/dse.log 2>&1