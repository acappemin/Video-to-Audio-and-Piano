export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file config.yaml train_example3.py