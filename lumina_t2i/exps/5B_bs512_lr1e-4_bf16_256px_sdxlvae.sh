#!/usr/bin/env sh

train_data_root='configs/data/JourneyDB.yaml'

model=DiT_Llama_5B_patch2
batch_size=512
lr=1e-4
precision=bf16
image_size=256
vae=sdxl

exp_name=${model}_bs${batch_size}_lr${lr}_${precision}_${image_size}px_vae${vae}
mkdir -p results/"$exp_name"

torchrun -—nproc-per-node=8 train.py \
    --master_port 18181 \
    --model ${model} \
    --data_path ${train_data_root} \
    --results_dir results/${exp_name} \
    --micro_batch_size 16 \
    --global_batch_size ${batch_size} --lr ${lr} \
    --data_parallel fsdp \
    --max_steps 3000000 \
    --ckpt_every 20000 --log_every 100 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --image_size ${image_size} \
    --vae ${vae} \
    2>&1 | tee -a results/"$exp_name"/output.log
