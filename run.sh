#!/bin/bash

pip install -e .

extract_data \
    --datapath /data/bodyct/experiments/luc_t10162/DRAGON \
    --task_ids 1 \
    --num_examples 0\
    --model_name qwen2.5:0.5b
