python src/export_model.py \
    --model_name_or_path /hy-tmp/models/qwen/Qwen2-7B-Instruct \
    --adapter_name_or_path /hy-tmp/checkpoints/pretrain/XX-XX-XX  \
    --template qwen \
    --finetuning_type lora \
    --export_dir /hy-tmp/checkpoints/pretrain/fine_tuned_model \
    --export_size 4 \
    --export_device auto