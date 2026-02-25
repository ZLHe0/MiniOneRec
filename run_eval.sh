#!/bin/bash
# Evaluation script configured for 4 GPUs (3,4,5,6)

# Set this to the model you want to evaluate (SFT or RL checkpoint)
EXP_NAME="./output/rl_qwen2.5-1.5b-instruct"

for category in "Industrial_and_Scientific"; do
    exp_name_clean=$(basename "$EXP_NAME")
    echo "Processing category: $category with model: $exp_name_clean"

    train_file=$(ls ./data/Amazon/train/${category}*.csv 2>/dev/null | head -1)
    test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)

    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi

    temp_dir="./temp/${category}-${exp_name_clean}"
    mkdir -p "$temp_dir"

    echo "Splitting test data..."
    python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "3,4,5,6"

    cudalist="3 4 5 6"
    echo "Starting parallel evaluation on 4 GPUs (3,4,5,6)..."
    for i in ${cudalist}; do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            echo "Starting evaluation on GPU $i"
            CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
                --base_model "$EXP_NAME" \
                --info_file "$info_file" \
                --category ${category} \
                --test_data_path "$temp_dir/${i}.csv" \
                --result_json_data "$temp_dir/${i}.json" \
                --batch_size 8 \
                --num_beams 50 \
                --max_new_tokens 256 \
                --temperature 1.0 \
                --guidance_scale 1.0 \
                --length_penalty 0.0 &
        fi
    done
    echo "Waiting for all evaluation processes..."
    wait

    output_dir="./results/${exp_name_clean}"
    mkdir -p "$output_dir"

    actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')
    echo "Merging results from GPUs: $actual_cuda_list"

    python ./merge.py \
        --input_path "$temp_dir" \
        --output_path "$output_dir/final_result_${category}.json" \
        --cuda_list "$actual_cuda_list"

    echo "Calculating metrics..."
    python ./calc.py \
        --path "$output_dir/final_result_${category}.json" \
        --item_path "$info_file"

    echo "Results saved to: $output_dir/final_result_${category}.json"
    echo "----------------------------------------"
done

echo "All done!"
