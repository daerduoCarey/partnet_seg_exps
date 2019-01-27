CUDA_VISIBLE_DEVICES=1 python pretrain.py \
    --category Chair \
    --level_id 3 \
    --model pretrain_model \
    --log_dir log_pretrain_model_Chair_3 \
    --epoch 151 \
    --batch 24 \
    --point_num 10000 \
    --learning_rate 1e-3

