"""
GRPO (Group Relative Policy Optimization) training script.
This is useful for aligning models using preference-based learning.
"""

import argparse
import yaml
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with TRL")
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
        help="Pretrained model name or path (ideally SFT'd first)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/grpo",
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


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load model and tokenizer
    # For GRPO, you typically start with an SFT'd model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    
    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Configure GRPO training
    # GRPO optimizes based on group preferences and relative rankings
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 5e-7),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 3),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        bf16=True,
        max_prompt_length=config.get("max_prompt_length", 512),
        max_completion_length=config.get("max_completion_length", 512),
        # GRPO-specific parameters
        num_generations=config.get("num_generations", 4),  # samples per prompt
        temperature=config.get("temperature", 0.9),
        kl_coef=config.get("kl_coef", 0.05),  # KL divergence penalty
        report_to="mlflow",  # MLflow tracking
        run_name=config.get("run_name", "grpo_experiment"),
    )
    
    # TODO: Load your dataset here
    # For GRPO, you need prompts and a reward function or preference data
    # train_dataset = load_your_grpo_data()
    train_dataset = None  # Replace with your data loading logic
    
    # You might also need a reward function depending on your setup
    # def reward_fn(samples):
    #     # Your reward calculation logic
    #     return rewards
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # reward_fn=reward_fn,  # uncomment if using a custom reward function
    )
    
    # Start training
    print(f"Starting GRPO training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
