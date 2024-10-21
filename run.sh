#!/bin/bash

pip install -e .

MODEL_NAME="phi3.5"
NUM_EXAMPLES=0
RUN_NAME="${MODEL_NAME}/${NUM_EXAMPLES}_examples"

extract_data \
    --task_id 1 \
    --model_name $MODEL_NAME \
    --num_examples $NUM_EXAMPLES \
    --run_name $RUN_NAME

evaluate \
    --task_ids 1 \
    --prediction_path /output/$RUN_NAME \
    --ground_truth_path /data/bodyct/experiments/luc_t10162/DRAGON/debug-test-set \
    --output_path /output/$RUN_NAME

