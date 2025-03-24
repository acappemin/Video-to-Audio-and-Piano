export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

accelerate launch --config_file config.yaml train_example3_4_2.py