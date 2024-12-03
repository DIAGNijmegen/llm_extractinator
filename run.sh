#!/bin/bash

pip install -e .

# for MODEL_NAME in "mistral-nemo" "llama3.1" "gemma2" "phi3" "llama3.2" "qwen2" "mixtral"
# do
for task_id in 25 26 27 28
do
BASE_DIR="/data/bodyct/experiments/luc_t10162/GitHub/LLM_data_extractor"
GROUND_TRUTH_PATH="/data/bodyct/experiments/luc_t10162/DRAGON/debug-test-set"
MODEL_NAME="mistral-nemo"
NUM_EXAMPLES=0
RUN_NAME="NER_debug/${MODEL_NAME}/${NUM_EXAMPLES}_examples"
PREDICTION_PATH="${BASE_DIR}/output/${RUN_NAME}"
OUTPUT_PATH="${PREDICTION_PATH}/metrics.json"

# extract_data \
#         --task_id 25 \
#         --model_name $MODEL_NAME \
#         --num_examples $NUM_EXAMPLES \
#         --run_name $RUN_NAME \
#         --max_context_len 8192

# post_process \
#     --output_path $PREDICTION_PATH \
#     --task_ids 25

# evaluate \
#     --task_ids 4 \
#     --prediction_path $PREDICTION_PATH \
#     --ground_truth_path $GROUND_TRUTH_PATH \
#     --output_path $OUTPUT_PATH



extract_data \
    --task_id $task_id \
    --model_name $MODEL_NAME \
    --num_examples $NUM_EXAMPLES \
    --run_name $RUN_NAME \
    --max_context_len 15000 \
    --num_predict 15000
done

# post_process \
#     --output_path $PREDICTION_PATH \
#     --task_ids 25 26 27

# evaluate \
#     --task_ids 25 26 27 \
#     --prediction_path $PREDICTION_PATH \
#     --ground_truth_path $GROUND_TRUTH_PATH \
#     --output_path $OUTPUT_PATH
# done


# for task_id in {25..28}
# do
#     extract_data \
#         --task_id $task_id \
#         --model_name $MODEL_NAME \
#         --num_examples $NUM_EXAMPLES \
#         --run_name $RUN_NAME \
#         --num_predict 8192 \
#         --max_context_len 8192
# done
# done
# post_process \
#     --output_path $PREDICTION_PATH \
#     --task_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

# evaluate \
#     --task_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
#     --prediction_path $PREDICTION_PATH \
#     --ground_truth_path $GROUND_TRUTH_PATH \
#     --output_path $OUTPUT_PATH

