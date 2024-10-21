#!/bin/bash

pip install -e .

MODEL_NAME="phi3.5"
NUM_EXAMPLES=0
RUN_NAME="$MODEL_NAME/$NUM_EXAMPLES"

extract_data \
    --task_id 1 \
    --model_name $MODEL_NAME \
    --num_examples $NUM_EXAMPLES \
    --run_name $RUN_NAME

evaluate \
    --task_ids 1 \
    --prediction_path /output/$RUN_NAME
