#!/bin/bash

pip install -e .

extract_data \
    --task_id 1 \
    --model_name phi3.5
