# Mini-reasoning

Training experiments using TRL for supervised fine-tuning and GRPO alignment.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### SFT (Supervised Fine-Tuning)

Train a model with standard next-token prediction:

```bash
./run_experiment.sh sft configs/sft_config.yaml
```

Or with a custom model:
```bash
./run_experiment.sh sft configs/sft_config.yaml meta-llama/Llama-3.2-3B
```

### GRPO (Group Relative Policy Optimization)

Train with preference-based learning (usually after SFT):

```bash
./run_experiment.sh grpo configs/grpo_config.yaml ./outputs/sft
```

## MLflow Tracking

Experiments are automatically tracked with MLflow. The `report_to="mlflow"` parameter in the training configs sends metrics to MLflow.

To view results:
```bash
mlflow ui
```

Then navigate to http://localhost:5000

## Configuration

Edit the YAML config files in `configs/` to adjust hyperparameters:
- `sft_config.yaml` - SFT training parameters
- `grpo_config.yaml` - GRPO training parameters

## Data Loading

The training scripts expect you to provide your own data loading logic. Look for the `TODO` comments in:
- `src/train_sft.py` - Load training dataset
- `src/train_grpo.py` - Load prompts and reward function