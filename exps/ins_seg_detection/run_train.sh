CUDA_VISIBLE_DEVICES=0 python3.5 train.py  \
    --model model \
    --log_dir log_Chair_level_3 \
    --category Chair \
    --level_id 3 \
    --num_ins 200 \
    --num_point 10000 \
    --max_epoch 251 \
    --batch_size 12 \
    --learning_rate 0.001 \
    --seg_loss_weight 1.0 \
    --ins_loss_weight 1.0 \
    --other_ins_loss_weight 1.0 \
    --l21_norm_loss_weight 0.1 \
    --conf_loss_weight 1.0 \
    --num_train_epoch_per_test 5 \
    --visu_dir visu \
    --visu_batch 2

