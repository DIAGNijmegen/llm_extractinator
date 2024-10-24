#!/bin/bash

pip install -e .

BASE_DIR="/data/bodyct/experiments/luc_t10162/GitHub/LLM_data_extractor"
GROUND_TRUTH_PATH="/data/bodyct/experiments/luc_t10162/DRAGON/debug-test-set"
MODEL_NAME="mistral-nemo"
NUM_EXAMPLES=0
RUN_NAME="${MODEL_NAME}/${NUM_EXAMPLES}_examples"
PREDICTION_PATH="${BASE_DIR}/output/${RUN_NAME}"
OUTPUT_PATH="${PREDICTION_PATH}/metrics.json"

extract_data \
    --task_id 14 \
    --model_name $MODEL_NAME \
    --num_examples $NUM_EXAMPLES \
    --run_name $RUN_NAME

evaluate \
    --task_ids 14 \
    --prediction_path $PREDICTION_PATH \
    --ground_truth_path $GROUND_TRUTH_PATH \
    --output_path $PREDICTION_PATH

