#!/bin/bash


# Lumina-Next supports any resolution (up to 2K)
# res="1024:024x1024 1536:1536x1536 1664:1664x1664 1792:1792x1792 2048:2048x2048"
res=1024:1024x1024
t=4
cfg=4.0
seed=25
steps=50
solver=euler
strength=0.6
image_dir=your/image/dir/here
model_dir=your/model/dir/here
cap_dir=your/caption/dir/here
out_dir=your/output/dir/here
python -u sample_img2img.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--image ${image_dir} \
--seed ${seed} \
--resolution ${res} \
--strength ${strength} \
--time_shifting_factor ${t} \
--cfg_scale ${cfg} \
--batch_size 1 \
--use_flash_attn True # You can set this to False if you want to disable the flash attention