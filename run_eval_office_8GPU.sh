#!/bin/bash
# Evaluation script for Office_Products, configured for 8 GPUs (0-7)

export CUDA_HOME=/work/envs/MiniOneRec

EXP_NAME=$1
if [ -z "$EXP_NAME" ]; then
    echo "Usage: bash run_eval_office_8GPU.sh <model_path>"
    exit 1
fi

for category in "Office_Products"; do
    exp_name_clean=$(basename "$EXP_NAME")
    echo "Processing category: $category with model: $exp_name_clean"

    test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)

    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi

    temp_dir="./temp/${category}-${exp_name_clean}"
    mkdir -p "$temp_dir"

    echo "Splitting test data..."
    python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "0,1,2,3,4,5,6,7"

    cudalist="0 1 2 3 4 5 6 7"
    echo "Starting parallel evaluation on 8 GPUs..."
    for i in ${cudalist}; do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
                --base_model "$EXP_NAME" \
                --info_file "$info_file" \
                --category ${category} \
                --test_data_path "$temp_dir/${i}.csv" \
                --result_json_data "$temp_dir/${i}.json" \
                --batch_size 8 \
                --num_beams 50 \
                --max_new_tokens 256 &
        fi
    done
    wait

    output_dir="./results/${exp_name_clean}"
    mkdir -p "$output_dir"

    actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')

    python ./merge.py \
        --input_path "$temp_dir" \
        --output_path "$output_dir/final_result_${category}.json" \
        --cuda_list "$actual_cuda_list"

    echo "Calculating metrics..."
    python ./calc.py \
        --path "$output_dir/final_result_${category}.json" \
        --item_path "$info_file"

    echo "Per-level analysis..."
    python ./calc_level.py \
        --path "$output_dir/final_result_${category}.json"
done

echo "All done!"
