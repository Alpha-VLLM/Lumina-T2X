#!/usr/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

python -u demo_music.py \
    --ckpt "/path/to/ckpt/music_generation" \
    --vocoder_ckpt "/path/to/ckpt/bigvnat" \
    --config_path "configs/lumina-text2music.yaml" \
    --sample_rate 16000
