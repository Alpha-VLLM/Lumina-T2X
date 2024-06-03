<p align="center">
 <img src="../assets/lumina-logo.png" width="40%"/>
 <br>
</p>

# Lumina-Next-T2I

The `Lumina-Next-T2I` model that uses Next-DiT with a 2B parameters model as well as using [Gemma-2B](https://huggingface.co/google/gemma-2b) as a text encoder. Compared with `Lumina-T2I`, it has faster inference speed, richer generation style, and more multilingual support, etc.

Our generative model has `Next-DiT` as the backbone, the text encoder is the `Gemma` 2B model, and the VAE uses a version of `sdxl` fine-tuned by stabilityai.

- Generation Model: Next-DiT
- Text Encoder: [Gemma-2B](https://huggingface.co/google/gemma-2b)
- VAE: [sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae)

## 📰 News

- [2024-5-12] 🚀🚀🚀 We release the next version of `Lumina-T2I`, called `Lumina-Next-T2I` for faster and lower memory usage image generation model.

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
pip install -e .
```

2. Prepare the pretrained checkpoints

⭐⭐ (Recommended) you can use `huggingface-cli` downloading our model:

```bash
huggingface-cli download --resume-download Alpha-VLLM/Lumina-Next-T2I --local-dir /path/to/ckpt
```

or using git for cloning the model you want to use:

```bash
git clone https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I
```

3. Converting `*.pth` files to `*.safetensors`

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

### CLI

1. Setting your personal inference configuration

Update your own personal inference settings to generate different styles of images, checking `config/infer/config.yaml` for detailed settings. Detailed config structure:

> `/path/to/ckpt` should be a directory containing `consolidated*.pth` and `model_args.pth`

```yaml
- settings:

  model:
    ckpt: "/path/to/ckpt"           # if ckpt is "", you should use `--ckpt` for passing model path when using `lumina` cli.
    ckpt_lm: ""                     # if ckpt is "", you should use `--ckpt_lm` for passing model path when using `lumina` cli.
    token: ""                       # if LLM is a huggingface gated repo, you should input your access token from huggingface and when token is "", you should `--token` for accessing the model.

  transport:
    path_type: "Linear"             # option: ["Linear", "GVP", "VP"]
    prediction: "velocity"          # option: ["velocity", "score", "noise"]
    loss_weight: "velocity"         # option: [None, "velocity", "likelihood"]
    sample_eps: 0.1
    train_eps: 0.2

  ode:
    atol: 1e-6                      # Absolute tolerance
    rtol: 1e-3                      # Relative tolerance
    reverse: false                  # option: true or false
    likelihood: false               # option: true or false

  infer:
      resolution: "1024x1024"     # option: ["1024x1024", "512x2048", "2048x512", "(Extrapolation) 1664x1664", "(Extrapolation) 1024x2048", "(Extrapolation) 2048x1024"]
      num_sampling_steps: 60      # range: 1-1000
      cfg_scale: 4.               # range: 1-20
      solver: "euler"             # option: ["euler", "dopri5", "dopri8"]
      t_shift: 4                  # range: 1-20 (int only)
      ntk_scaling: true           # option: true or false
      proportional_attn: true     # option: true or false
      seed: 0                     # rnage: any number
```

- model:
  - `ckpt`: lumina-t2i checkpoint path from [huggingface repo](https://huggingface.co/Alpha-VLLM/Lumina-T2I) containing `consolidated*.pth` and `model_args.pth`.
  - `ckpt_lm`: LLM checkpoint.
  - `token`: huggingface access token for accessing gated repo.
- transport:
  - `path_type`: the type of path for transport: 'Linear', 'GVP' (Geodesic Vector Pursuit), or 'VP' (Vector Pursuit).
  - `prediction`: the prediction model for the transport dynamics.
  - `loss_weight`: the weighting of different components in the loss function, can be 'velocity' for dynamic modeling, 'likelihood' for statistical consistency, or None for no weighting
  - `sample_eps`: sampling in the transport model.
  - `train_eps`: training to stabilize the learning process.
- ode:
  - `atol`: Absolute tolerance for the ODE solver. (options: ["Linear", "GVP", "VP"])
  - `rtol`: Relative tolerance for the ODE solver. (option: ["velocity", "score", "noise"])
  - `reverse`: run the ODE solver in reverse. (option: [None, "velocity", "likelihood"])
  - `likelihood`: Enable calculation of likelihood during the ODE solving process.
- infer:
  - `resolution`: generated image resolution.
  - `num_sampling_steps`: sampling step for generating image.
  - `cfg_scale`: classifier-free guide scaling factor
  - `solver`: solver for image generation.
  - `t_shift`: time shift factor.
  - `ntk_scaling`: ntk rope scaling factor.
  - `proportional_attn`: Whether to use proportional attention.
  - `seed`: random initialization seeds.

2. Run with CLI

inference command:
```bash
lumina_next infer -c <config_path> <caption_here> <output_dir>
```

e.g. Demo command:

```bash
cd lumina_next_t2i
lumina_next infer -c "configs/infer/settings.yaml" "a snow man of ..." "./outputs"
```
