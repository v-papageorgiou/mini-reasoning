"""
Supervised fine-tuning script using TRL.
Trains a model with standard next-token prediction.
"""

import argparse
import yaml
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator


def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT training with TRL")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config YAML file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/sft",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name to download and use for training"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

    

def load_dataset(dataset_name):
    if dataset_name == "open-thoughts/OpenThoughts-114k":
        from datasets import load_dataset
        
        def format_dataset_openthoughts(example):
            # Combine the conversations into a single text input
            return {
                "prompt": [{"role": "user", "content": example["problem"]}],
                "completion": [
                    {"role": "assistant", "content": f"<think>{example['deepseek_reasoning']}</think><answer>{example['deepseek_solution']}</answer>"}
                ]
            }
        dataset = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")
        # Keep only the samples that have "domain = math"
        dataset = dataset.filter(lambda x: x["domain"] == "math")
        dataset = dataset.map(format_dataset_openthoughts)
        return dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize accelerator for distributed setup if needed
    accelerator = Accelerator()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        # device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    
    # Some models don't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Configure SFT training
    # The report_to parameter handles MLflow integration automatically
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-5),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 3),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        bf16=True,  # use bfloat16 for better numerical stability
        # max_seq_length=config.get("max_seq_length", 512),
        packing=config.get("packing", False),  # pack multiple samples into one sequence
        loss_type=config.get("loss_type", "sft"),  # can be "sft" or "dft"
        report_to="mlflow",  # this enables MLflow tracking
        run_name=config.get("run_name", "sft_experiment"),
    )

    # Load dataset
    train_dataset = load_dataset(args.dataset_name)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Start training
    print(f"Starting SFT training with {len(train_dataset)} examples...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
