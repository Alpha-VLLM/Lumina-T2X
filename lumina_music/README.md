# Lumina Text-to-Music

We will provide our implementation and pretrained models as open source in this repository recently.

- Generation Model: Flag-DiT
- Text Encoder: [FLAN-T5-Large](https://huggingface.co/google/flan-t5-large)
- VAE: Make an Audio 2, finetuned from [Makee an Audio](https://github.com/Text-to-Audio/Make-An-Audio)
- Decoder: [Vocoder](https://github.com/NVIDIA/BigVGAN)
- `Lumina-T2Music` Checkpoints: [huggingface](https://huggingface.co/Alpha-VLLM/Lumina-T2Music)

## 📰 News

- [2024-06-07] 🚀🚀🚀 We release the initial version of `Lumina-T2Music` for text-to-music generation.

## Installation

Before installation, ensure that you have a working ``nvcc``

```bash
# The command should work and show the same version number as in our case. (12.1 in our case).
nvcc --version
```

On some outdated distros (e.g., CentOS 7), you may also want to check that a late enough version of
``gcc`` is available

```bash
# The command should work and show a version of at least 6.0.
# If not, consult distro-specific tutorials to obtain a newer version or build manually.
gcc --version
```

Downloading Lumina-T2X repo from github:

```bash
git clone https://github.com/Alpha-VLLM/Lumina-T2X
```

### 1. Create a conda environment and install PyTorch

Note: You may want to adjust the CUDA version [according to your driver version](https://docs.nvidia.com/deploy/cuda-compatibility/#default-to-minor-version).

```bash
conda create -n Lumina_T2X -y
conda activate Lumina_T2X
conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 2. Install dependencies

>[!Warning]
> The environment dependencies for Lumina-T2Music are different from those for Lumina-T2I. Please install the appropriate environment.

Installing `Lumina-T2Music` dependencies:

```bash
cd .. # If you are in the `lumina_music` directory, execute this line.
pip install -e ".[music]"
```

or you can use `requirements.txt` to install the environment.

```bash
cd lumina_music # If you are not in the `lumina_music` folder, run this line.
pip install -r requirements.txt
```

### 3. Install ``flash-attn``

```bash
pip install flash-attn --no-build-isolation
```

### 4. Install [nvidia apex](https://github.com/nvidia/apex) (optional)

>[!Warning]
> While Apex can improve efficiency, it is *not* a must to make Lumina-T2X work.
>
> Note that Lumina-T2X works smoothly with either:
> + Apex not installed at all; OR
> + Apex successfully installed with CUDA and C++ extensions.
>
> However, it will fail when:
> + A Python-only build of Apex is installed.
>
> If the error `No module named 'fused_layer_norm_cuda'` appears, it typically means you are using a Python-only build of Apex. To resolve this, please run `pip uninstall apex`, and Lumina-T2X should then function correctly.

You can clone the repo and install following the official guidelines (note that we expect a full
build, i.e., with CUDA and C++ extensions)

```bash
pip install ninja
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Inference

### Preparation

Prepare the pretrained checkpoints.

⭐⭐ (Recommended) you can use `huggingface-cli` downloading our model:

```bash
huggingface-cli download --resume-download Alpha-VLLM/Lumina-T2Music --local-dir /path/to/ckpt
```

or using git for cloning the model you want to use:

```bash
git clone https://huggingface.co/Alpha-VLLM/Lumina-T2Music
```

### Web Demo

To host a local gradio demo for interactive inference, run the following command:

1. updated `AutoencoderKL` ckpt path

you should update `configs/lumina-text2music.yaml` to set `AutoencoderKL` checkpoint path. Please replace `/path/to/ckpt` with the path where your checkpoints are located (<real_path>).

```diff
  ...
        depth: 16
        max_len: 1000

    first_stage_config:
      target: models.autoencoder1d.AutoencoderKL
      params:
        embed_dim: 20
        monitor: val/rec_loss
        - ckpt_path: /path/to/ckpt/maa2/maa2.ckpt
        + ckpt_path: <real_path>/maa2/maa2.ckpt
        ddconfig:
          double_z: true
          in_channels: 80
          out_ch: 80
  ...
```

2. setting `Lumina-T2Music` and `Vocoder` checkpoint path and run demo

Please replace `/path/to/ckpt` with the actual downloaded path.

```bash
# `/path/to/ckpt` should be a directory containing `music_generation`, `maa2`, and `bigvnat`.

# default
python -u demo_music.py \
    --ckpt "/path/to/ckpt/music_generation" \
    --vocoder_ckpt "/path/to/ckpt/bigvnat" \
    --config_path "configs/lumina-text2music.yaml" \
    --sample_rate 16000
```

or you can run `run_music.sh` script for web demo after updating `AutoencoderKL` ckpt path on `configs/lumina-text2music.yaml`, `--ckpt`, and `--vocoder_ckpt`.

```bash
bash run_music.sh
```

## Disclaimer

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
