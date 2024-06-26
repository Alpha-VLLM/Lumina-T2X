<p align="center">
 <img src="../assets/lumina-logo.png" width="40%"/>
 <br>
</p>

# Lumina-Next-T2I Compositional Generation

`Lumina-Next-T2I Compositional Generation` is the multi-caption version of `Lumina-Next-T2I`. You don't need to retrain the `Lumina-Next-T2I` model; simply adjust the attention calculation method.

Our generative model has `Next-DiT` as the backbone, the text encoder is the `Gemma` 2B model, and the VAE uses a version of `sdxl` fine-tuned by stabilityai.

- Generation Model: Next-DiT
- Text Encoder: [Gemma-2B](https://huggingface.co/google/gemma-2b)
- VAE: [sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae)

## 📰 News

**[2024-06-03] 🔥🔥🔥 We have released the `Compositional Generation` version of `Lumina-Next-T2I`, which enables compositional generation with multiple caption for different region. [model](https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I). [DEMO](http://106.14.2.150:10023/)**

## 🎮 Model Zoo

More checkpoints of our model will be released soon~

| Resolution | Next-DiT Parameter| Text Encoder | Prediction | Download URL  |
| ---------- | ----------------------- | ------------ | -----------|-------------- |
| 1024       | 2B             |    [Gemma-2B](https://huggingface.co/google/gemma-2b)  |   Rectified Flow | [hugging face](https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I) |

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

  ```bash
  pip install diffusers fairscale accelerate tensorboard transformers gradio torchdiffeq click
  ```

  or you can use

  ```bash
  cd lumina_next_t2i
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

To ensure that our generative model is ready to use right out of the box, we provide a user-friendly CLI program and a locally deployable Web Demo site.

### Preparation

1. Install Lumina-T2I

```bash
cd .. # go to the root of the repo.
pip install -e .
```

2. Prepare the pretrained checkpoints

⭐⭐ (Recommended) you can use `huggingface-cli` downloading our model:

```bash
huggingface-cli download --resume-download Alpha-VLLM/Lumina-Next-SFT --local-dir /path/to/ckpt
```

or using git for cloning the model you want to use:

```bash
git clone https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT
```

>[!Note]
> For Chinese user using command below to download the model:
> ```bash
> git lfs install
> git clone https://www.wisemodel.cn/Alpha-VLLM/Lumina-Next-SFT.git
> ```

3. Converting `*.pth` files to `*.safetensors`

>[!Note]
> If you already have a model in safetensors format, you can skip this step.

If you are loading your own trained model, please convert it to `.safetensors` first for security reasons before loading. Assuming your trained model path is `/path/to/your/own/model.pth` and your save directory is `/path/to/new/model`.

```bash
lumina_next convert "/path/to/your/own/model.pth" "/path/to/new/directory/" # convert to `.safetensors`
```

Explanation of the `lumina convert` command:
```bash
# <weight_path> means your trained model path.
# <output_dir> means the directory where you want to save the model.
lumina_next convert <weight_path> <output_dir>

# example 1:
lumina_next convert "/path/to/your/own/model.pth" "/path/to/new/directory/" # convert to `.safetensors`

# example 2:
lumina_next convert "/path/to/your/own/model.safetensors" "/path/to/new/directory/" # convert to `.pth`
```

### Web Demo

To host a local gradio demo for interactive inference, run the following command:

```bash
# `/path/to/ckpt` should be a directory containing `consolidated*.pth` and `model_args.pth`

# default
python -u demo.py --ckpt "/path/to/ckpt"

# the demo by default uses bf16 precision. to switch to fp32:
python -u demo.py --ckpt "/path/to/ckpt" --precision fp32

# use ema model
python -u demo.py --ckpt "/path/to/ckpt" --ema
```
