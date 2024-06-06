#!/usr/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

python demo_music.py \
    --ckpt "</path/to/music/generation>" \
    --vocoder_ckpt "</path/to/vocoder>" \
    --config_path configs/lumina-text2music.yaml \
    --sample_rate 16000
