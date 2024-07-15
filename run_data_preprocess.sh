# 对GPT生成的结果进行微调，处理训练集中的剩余数据
python ./src/main.py \
    --method random \
    --is_process_data \
    --selected_k 3\
    --prefix /root/LaiDA/preprocess_data/ \
    --inst_prefix /root/LaiDA/preprocess_data/Instruction/ \
    --results_path_prefix /root/LaiDA/preprocess_data/results/ \
    --do_train \
    --seed 42 \
    --dataset_dir ./preprocess_data \
    --model_name_or_path /hy-tmp/models/qwen/Qwen2-7B-Instruct \
    --finetuning_type lora \
    --template qwen \
    --lora_target all \
    --lora_rank 32 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --logging_steps 20 \
    --save_steps 400 \
    --val_size 0.1 \
    --learning_rate 8e-5 \
    --num_train_epochs 3.0 \
    --bf16 \
#    --output_dir /hy-tmp/checkpoints/data_preprocess/XX-XX-XX