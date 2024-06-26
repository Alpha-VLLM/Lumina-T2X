<p align="center">
 <img src="../assets/lumina-logo.png" width="40%"/>
 <br>
</p>

# Lumina-Next-T2I-Mini

`Lumina-Next-T2I-Mini` is a simplified version of `Lumina-Next-T2I`, with
1. trimmed `transport` module; and
2. removed model-parallel stuff (which exists but makes no effect in the `Lumina-Next-T2I` project)

Though simplified, this directory retains **ALL** the functionalities that were *actually used* by us during the training and inference of Lumina-Next-T2I.

Currently, this codebase also supports the training (including Dreambooth) and inference of **Stable Diffusion 3** (SD3). Since both SD3 and Lumina are flow-based diffusion transformers, we integrated SD3 with minimal changes based on our Lumina codebase.

## 🎮 Model Zoo

More checkpoints of our model will be released soon~

| Model | Resolution | Next-DiT Parameter| Text Encoder | Download URL  |
| ---------- | ---------- | ----------------------- | ------------ |-------------- |
| Lumina-Next-T2I | 1024       | 2B             |    [Gemma-2B](https://huggingface.co/google/gemma-2b)  | [hugging face](https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I) |
| Lumina-Next-SFT | 1024       | 2B             |    [Gemma-2B](https://huggingface.co/google/gemma-2b)  | [hugging face](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT) |

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
  pip install diffusers accelerate tensorboard transformers gradio torchdiffeq click
  ```

  or you can use

  ```bash
  pip install -r requirements.txt
  ```

### 3. Install ``flash-attn``

  To speed up the training and inference, we recommend installing the ``flash-attn`` package. If you have trouble installing it, you can skip this step and pass the argument ``--use_flash_attn False`` to the training and inference scripts.
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

### Prepare Checkpoints

#### 1. Download pretrained checkpoints

⭐⭐ (Recommended) you can use huggingface_cli downloading our model:

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

#### 2. Use checkpoints trained by yourself

If you are loading your own trained model, please convert `*.pth` files to `.safetensors` first for security reasons before loading. Assuming your trained model path is `/path/to/your/own/model.pth` and your save directory is `/path/to/new/model`.

```bash
lumina_next convert "/path/to/your/own/model.pth" "/path/to/new/directory/" # convert to `.safetensors`
```

Explanation of the `lumina_next convert` command:
```bash
# <weight_path> means your trained model path.
# <output_dir> means the directory where you want to save the model.
lumina_next convert <weight_path> <output_dir>

# example 1:
lumina_next convert "/path/to/your/own/model.pth" "/path/to/new/directory/" # convert to `.safetensors`

# example 2:
lumina_next convert "/path/to/your/own/model.safetensors" "/path/to/new/directory/" # convert to `.pth`
```

### Direct Inference
To generate images directly from the inference code (for development), run the following script with customized arguments:

```bash
# For Lumina
bash scripts/sample.sh

# For Lumina img2img translation
bash scripts/sample_img2img.sh

# For SD3
bash scripts/sample_sd3.sh
```
You can personalize more arguments by checking the `sample.py` and `sample_sd3.py` file.

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

# disable the use Flash Attention
python -u demo.py --ckpt "/path/to/ckpt" --use_flash_attn False
```
