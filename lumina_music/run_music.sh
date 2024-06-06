#!/usr/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

python demo_music.py \
    --ckpt "</path/to/music/generation>" \
    --vocoder_ckpt "</path/to/vocoder>" \
    --config_path configs/lumina-text2music.yaml \
    --sample_rate 16000
    
# TRITON_PTXAS_PATH="/usr/local/cuda-11.7/bin/ptxas" python3 flow.py \
#     --outdir output_dir \
#     -r ../checkpoints_tmp/music_generation/119.ckpt  \
#     -b configs/lumina-text2music.yaml \
#     --scale 5.0 \
#     --vocoder-ckpt ../checkpoints_tmp/bigvnat \
#     --test-dataset audiocaps