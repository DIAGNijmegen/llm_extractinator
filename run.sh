#!/bin/bash

pip install -e .

extract_data \
    --data_path /data/bodyct/experiments/luc_t10162/DRAGON \
    --ground_truth_path /data/bodyct/experiments/luc_t10162/DRAGON/debug-test-set \
    --task_ids 1 \
    --num_examples 0\
    --model_name qwen2.5:1.5b
