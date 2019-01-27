CUDA_VISIBLE_DEVICES=0 python train.py \
    --category Chair \
    --level_id 3 \
    --model model \
    --log_dir log_finetune_model_Chair_3 \
    --epoch 151 \
    --batch 1 \
    --point_num 10000 \
    --group_num 200 \
    --restore_dir log_pretrain_model_Chair_3 \
    --margin_same 1 \
    --margin_diff 2

