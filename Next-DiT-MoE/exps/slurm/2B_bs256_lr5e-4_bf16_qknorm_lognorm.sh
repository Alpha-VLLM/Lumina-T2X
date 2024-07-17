#!/usr/bin/env sh

train_data_root='/path/to/imagenet/images/train'

model=DiT_Llama_2B_patch2
batch_size=256
lr=5e-4
precision=bf16

exp_name=${model}_bs${batch_size}_lr${lr}_${precision}_qknorm
mkdir -p results/"$exp_name"

python -u train.py \
    --model ${model} \
    --data_path ${train_data_root} \
    --results_dir results/"$exp_name" \
    --micro_batch_size 32 \
    --global_batch_size ${batch_size} --lr ${lr} \
    --data_parallel sdp \
    --max_steps 3000000 \
    --ckpt_every 10000 --log_every 100 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --snr_type "lognorm" \
    2>&1 | tee -a results/"$exp_name"/output.log
