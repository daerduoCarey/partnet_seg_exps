CUDA_VISIBLE_DEVICES=0 python valid.py \
    --model model \
    --category Chair \
    --level_id 3 \
    --num_ins 200 \
    --log_dir log_finetune_model_Chair_3 \
    --valid_dir valid \
    --num_point 10000 \
    --batch_size 1 \
    --margin_same 1.0
