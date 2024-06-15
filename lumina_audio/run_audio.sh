#!/usr/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

python -u demo_audio.py \
    --ckpt "/path/to/ckpt/audio_generation" \
    --vocoder_ckpt "/path/to/ckpt/bigvnat" \
    --config_path "configs/lumina-text2audio.yaml" \
    --sample_rate 16000
