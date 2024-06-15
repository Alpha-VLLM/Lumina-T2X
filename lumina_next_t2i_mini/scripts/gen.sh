#!/bin/bash

# export HF_HOME="/data4/zl/.cache/huggingface"
# export HF_DATASETS_CACHE="/data4/zl/.cache/huggingface/datasets"
# export TRANSFORMERS_CACHE="/data4/zl/.cache/huggingface/models"

# CUDA_VISIBLE_DEVICES=2 python -u demo.py --ckpt /home/pgao/zl/zl/ckpt/lumina-next-hq-v2/0004000 --ema
# res="1024:1024x1024 1536:1536x1536 1664:1664x1664 1792:1792x1792 2048:2048x2048"
res=1024:1024x1024
# res=3072:3072x3072
t=6
cfg=7.0
seed=25
steps=10
CUDA_VISIBLE_DEVICES=1 python -u sample.py --ckpt /home/pgao/zl/data3/zl/ckpt/lumina-next/0106000 \
--image_save_path samples/test_sd3_sft6 \
--solver midpoint --num_sampling_steps ${steps} \
--caption_path captions_v3.txt \
--seed ${seed} \
--resolution ${res} \
--time_shifting_factor ${t} \
--cfg_scale ${cfg} \
--batch_size 1 \


# CUDA_VISIBLE_DEVICES=3 python -u sample_autoguide.py \
# --ckpt "/home/pgao/zl/zl/ckpt/lumina-next-hq/0070000" \
# --student_ckpt "/home/pgao/zl/zl/ckpt/lumina-next/0040000" \
# --image_save_path samples/v6_70k_autoguide_v5_40k_${cfg}_inter_v3 \
# --sampling-method midpoint --num_sampling_steps ${steps} \
# --caption_path captions_v3.txt \
# --seed ${seed} \
# --resolution ${res} \
# --time_shifting_factor ${t} \
# --cfg_scale ${cfg} \
# --batch_size 1 \
