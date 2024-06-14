#!/bin/bash


# SD3 supports up to 1.5K resolution
# res="1024:024x1024 1536:1536x1536"
res=1024:1024x1024
t=1
cfg=7.0
seed=25
steps=20
solver=midpoint
model_dir=stabilityai/stable-diffusion-3-medium-diffusers
cap_dir=your/caption/dir/here
out_dir=your/output/dir/here
python -u sample.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--seed ${seed} \
--resolution ${res} \
--time_shifting_factor ${t} \
--cfg_scale ${cfg} \
--batch_size 1 \