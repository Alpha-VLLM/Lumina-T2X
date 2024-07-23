#!/bin/bash

# res=1024:1024x1024
# res=2048:2048x2048
# res=2560:2560x2560
res=4096:4096x4096
t=7
cfg=8.0
seed=17
steps=30
model_dir=path_to_your_ckpt
n=10

CUDA_VISIBLE_DEVICES=1 \
python generate.py --name flip.campfire.man \
    --prompts "people at a campfire" "an old man"\
    --style "an oil painting of" \
    --views identity flip  \
    --num_samples ${n} \
    --num_inference_steps ${steps} \
    --ckpt ${model_dir} \
    --seed ${seed} \
    --resolution ${res} \
    --time_shifting_factor ${t} \
    --cfg_scale ${cfg} \
    --batch_size 1 \
    --use_flash_attn True # You can set this to False if you want to disable the flash attention
