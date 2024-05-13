<p align="center">
 <img src="../assets/lumina-logo.png" width="40%"/> 
 <br>
</p>

# Lumina-T2I

Lumina-T2I is a model that generates images base on text condition, supporting various text encoders and models of different parameter sizes. With minimal training costs, it achieves high-quality image generation by training from scratch. Additionally, it offers usage through CLI console programs and Web Demo displays.

Our generative model has `Large-DiT` as the backbone, the text encoder is the `LLaMA2` 7B model, and the VAE uses a version of `sdxl` fine-tuned by stabilityai.

- Generation Model: Large-DiT
- Text Encoder: [LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- VAE: [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae)

## 📰 News

- [2024-4-1] 🚀🚀🚀 We release the initial version of Lumina-T2I for text-to-image generation

## 🎮 Model Zoo

More checkpoints of our model will be released soon~

| Resolution | Flag-DiT Parameter| Text Encoder | Prediction | Download URL  |
| ---------- | ----------------------- | ------------ | -----------|-------------- |
| 1024       | 5B             |    [LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |   Rectified Flow | [hugging face](https://huggingface.co/Alpha-VLLM/Lumina-T2I) |

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
  cd lumina-t2i
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

## Training

> [!Warning]
> **Lumina-T2I employs FSDP for training large diffusion models. FSDP shards parameters, optimizer states, and gradients across GPUs.
> Thus, at least 8 GPUs are required for full fine-tuning of the Lumina-T2X 5B model.
> Parameter-efficient Finetuning of Lumina-T2X shall be released soon.**

## Inference

To ensure that our generative model is ready to use right out of the box, we provide a user-friendly CLI program and a locally deployable Web Demo site.

### CLI

1. Install Lumina-T2I

```bash
pip install -e .
```

2. Prepare the pretrained model

⭐⭐ (Recommended) you can use huggingface_cli downloading our model:

```bash
huggingface-cli download --resume-download Alpha-VLLM/Lumina-T2I --local-dir /path/to/ckpt
```

or using git for cloning the model you want to use:

```bash
git clone https://huggingface.co/Alpha-VLLM/Lumina-T2I
``` 

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

  sde:
    sampling_method: "Euler"        # option: ["Euler", "Heun"]
    diffusion_form: "sigma"         # option: ["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"]
    diffusion_norm: 1.0             # range: 0-1
    last_step: Mean                 # option: [None, "Mean", "Tweedie", "Euler"]
    last_step_size: 0.04

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
- sde
  - `sampling-method`: the numerical method used for sampling the stochastic differential equation: 'Euler' for simplicity or 'Heun' for improved accuracy.
  - `diffusion-form`: form of diffusion coefficient in the SDE
  - `diffusion-norm`: Normalizes the diffusion coefficient, affecting the scale of the stochastic component.
  - `last-step`: form of last step taken in the SDE
  - `last-step-size`: size of the last step taken
- infer
  - `resolution`: generated image resolution.
  - `num_sampling_steps`: sampling step for generating image.
  - `cfg_scale`: classifier-free guide scaling factor
  - `solver`: solver for image generation.
  - `t_shift`: time shift factor.
  - `ntk_scaling`: ntk rope scaling factor.
  - `proportional_attn`: Whether to use proportional attention.
  - `seed`: random initialization seeds.

1. Run with CLI

inference command:
```bash
lumina infer -c <config_path> <caption_here> <output_dir>
```

e.g. Demo command:

```bash
cd lumina_t2i
lumina infer -c "config/infer/settings.yaml" "a snow man of ..." "./outputs"
```

### Web Demo

To host a local gradio demo for interactive inference, run the following command:

```bash
# `/path/to/ckpt` should be a directory containing `consolidated*.pth` and `model_args.pth`

# default
python -u demo.py ODE --ckpt "/path/to/ckpt"

# the demo by default uses bf16 precision. to switch to fp32:
python -u demo.py ODE --ckpt "/path/to/ckpt" --precision fp32 

# use ema model
python -u demo.py ODE --ckpt "/path/to/ckpt" --ema
```
