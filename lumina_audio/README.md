# Lumina Text-to-Audio

`Lumina Text-to-Audio` is a music generation model developed based on FlagDiT. It uses [T5-v1.1-XXL](https://huggingface.co/google/t5-v1_1-xxl) as the text encoder and [Vocoder](https://github.com/NVIDIA/BigVGAN) as the decoder.

>[!Warning]
>The current version of Lumina Text-to-Audio requires the use of structure caption for audio generation. We will soon release a version that does not require structure caption.

- Generation Model: Flag-DiT
- Text Encoder: [T5-v1.1-XXL](https://huggingface.co/google/t5-v1_1-xxl)
- VAE: Make an Audio 2, finetuned from [Make an Audio](https://github.com/Text-to-Audio/Make-An-Audio)
- Decoder: [Vocoder](https://github.com/NVIDIA/BigVGAN)
- `Lumina-T2Audio` Checkpoints: [huggingface](https://huggingface.co/Alpha-VLLM/Lumina-T2Audio)

## 📰 News

- [2024-06-19] 🚀🚀🚀 We release the initial version of `Lumina-T2Audio` for text-to-audio generation.

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
> The environment dependencies for Lumina-T2Audio are different from those for Lumina-T2I. Please install the appropriate environment.

Installing `Lumina-T2Audio` dependencies:

```bash
cd .. # If you are in the `lumina_audio` directory, execute this line.
pip install -e ".[audio]"
```

or you can use `requirements.txt` to install the environment.

```bash
cd lumina_audio # If you are not in the `lumina_audio` folder, run this line.
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
huggingface-cli download --resume-download Alpha-VLLM/Lumina-T2Audio --local-dir /path/to/ckpt
```

or using git for cloning the model you want to use:

```bash
git clone https://huggingface.co/Alpha-VLLM/Lumina-T2Audio
```

### Web Demo

To host a local gradio demo for interactive inference, run the following command:

1. updated `AutoencoderKL` ckpt path

you should update `configs/lumina-text2audio.yaml` to set `AutoencoderKL` checkpoint path. Please replace `/path/to/ckpt` with the path where your checkpoints are located (<real_ckpt_path>).

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
        + ckpt_path: <real_ckpt_path>/maa2/maa2.ckpt
        ddconfig:
          double_z: true
          in_channels: 80
          out_ch: 80
  ...
    cond_stage_config:
      target: models.encoders.modules.FrozenCLAPFLANEmbedder
      params:
        - weights_path: /path/to/ckpt/CLAP/CLAP_weights_2022.pth
        + weights_path: <real_ckpt_path>/CLAP/CLAP_weights_2022.pth

```

2. setting `Lumina-T2Audio` and `Vocoder` checkpoint path and run demo

Please replace `/path/to/ckpt` with the actual downloaded path.

```bash
# `/path/to/ckpt` should be a directory containing `audio_generation`, `maa2`, and `bigvnat`.

# default
python -u demo_audio.py \
    --ckpt "/path/to/ckpt/audio_generation" \
    --vocoder_ckpt "/path/to/ckpt/bigvnat" \
    --config_path "configs/lumina-text2audio.yaml" \
    --sample_rate 16000
```

or you can run `run_audio.sh` script for web demo after updating `AutoencoderKL` ckpt path on `configs/lumina-text2audio.yaml`, and updating `--ckpt`, and `--vocoder_ckpt` on `run_audio.sh`.

3. setting openai api key for generating structure caption.

Please replace the line in `n2s_openai.py`:

```diff
- openai_key = 'your openai api key here'
+ openai_key = '<your real openai api key>'
```

If you have other relay station APIs, please modify the `base_url` accordingly. The default setting uses OpenAI's `base_url`.

```diff
- base_url = ""
+ base_url = "<your base url>"
```

4. running the demo

```bash
bash run_audio.sh
```

## Disclaimer

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
