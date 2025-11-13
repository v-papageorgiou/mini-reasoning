#!/bin/bash

# Usage: ./eval.sh [sft|dft]
# Example: ./eval.sh sft

if [ -z "$1" ]; then
    echo "Error: Please specify model type (sft or dft)"
    echo "Usage: $0 [sft|dft]"
    exit 1
fi

MODEL_TYPE=$1

if [ "$MODEL_TYPE" != "sft" ] && [ "$MODEL_TYPE" != "dft" ]; then
    echo "Error: Invalid model type. Must be 'sft' or 'dft'"
    echo "Usage: $0 [sft|dft]"
    exit 1
fi


python -m eval.eval \
    --model hf \
    --tasks MATH500 \
    --model_args "pretrained=/home/vpapageorgio/mini-reasoning/outputs/${MODEL_TYPE}," \
    --batch_size 8 \
    --output_path logs_${MODEL_TYPE}/