MODEL_TYPE="pi05_baseline"

export WANDB_API_KEY=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

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_baseline \
--exp-name=${MODEL_TYPE}_your_model_name \
--batch-size=64 \
--num-workers=4 \
--fsdp-devices=4 \
--dataset-path=data/robomme_preprocessed_data