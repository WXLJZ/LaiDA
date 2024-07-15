# 对构造的训练集进行微调
python ./src/main.py \
    --method gnn \
    --selected_k 3\
    --bert_path /hy-tmp/models/bert-base-chinese \
    --gnn_path /root/LaiDA/gnn_checkpoints/best_gnn_model.pt \
    --prefix /root/LaiDA/Fine_tuning_data/ \
    --inst_prefix /root/LaiDA/Fine_tuning_data/instruction/ \
    --results_path_prefix /root/LaiDA/results/ \
    --do_train \
    --seed 42 \
    --dataset_dir ./Fine_tuning_data \
    --model_name_or_path /hy-tmp/checkpoints/pretrained/fine_tuned_model \
    --finetuning_type lora \
    --template qwen \
    --lora_target all \
    --lora_rank 32 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --logging_steps 800 \
    --save_steps 800 \
    --val_size 0.1 \
    --learning_rate 8e-5 \
    --num_train_epochs 3.0 \
    --bf16 \
#    --output_dir /hy-tmp/checkpoints/gnn_XX-XX-XX