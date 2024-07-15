
python src/gnnencoder/train.py \
    --save_path ./gnn_checkpoints/ \
    --bert_model_path /hy-tmp/models/bert-base-chinese \
    --data_path /root/LaiDA/Fine_tuning_data/Fine_tuning_train.json \
    --batch_size 256 \
    --max_len 128 \
    --lig_top_k 0.2 \
    --epoch_num 12 \
    --lr 1e-4