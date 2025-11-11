#!/bin/bash
# Quick launcher for running experiments

EXPERIMENT_TYPE=$1
CONFIG_PATH=$2
DATASET_NAME=$3
MODEL_NAME=${4:-"meta-llama/Llama-3.2-1B"}

if [ -z "$EXPERIMENT_TYPE" ] || [ -z "$CONFIG_PATH" ] || [ -z "$DATASET_NAME" ]; then
    echo "Usage: ./run_experiment.sh [sft|dft|grpo] <config_path> <dataset_name> [model_name]"
    echo "Example: ./run_experiment.sh sft configs/sft_config.yaml open-thoughts/OpenThoughts-114k Qwen/Qwen2.5-0.5B"
    exit 1
fi

# Make sure we're using the right accelerate config
export ACCELERATE_CONFIG_FILE="configs/accelerate/single_gpu.yaml"

if [ "$EXPERIMENT_TYPE" == "sft" ]; then
    echo "Running SFT experiment..."
    accelerate launch --config_file $ACCELERATE_CONFIG_FILE src/train_sft.py \
        --config $CONFIG_PATH \
        --dataset_name $DATASET_NAME \
        --model_name $MODEL_NAME
elif [ "$EXPERIMENT_TYPE" == "dft" ]; then
    echo "Running DFT experiment..."
    accelerate launch --config_file $ACCELERATE_CONFIG_FILE src/train_sft.py \
        --config $CONFIG_PATH \
        --dataset_name $DATASET_NAME \
        --model_name $MODEL_NAME \
        --output_dir ./outputs/dft
elif [ "$EXPERIMENT_TYPE" == "grpo" ]; then
    echo "Running GRPO experiment..."
    accelerate launch --config_file $ACCELERATE_CONFIG_FILE src/train_grpo.py \
        --config $CONFIG_PATH \
        --dataset_name $DATASET_NAME \
        --model_name $MODEL_NAME
else
    echo "Unknown experiment type: $EXPERIMENT_TYPE"
    echo "Must be either 'sft', 'dft', or 'grpo'"
    exit 1
fi
