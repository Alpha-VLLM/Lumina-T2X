<p align="center">
 <img src="../assets/lumina-logo.png" width="40%"/> 
 <br>
</p>

# Lumina-T2I

Lumina-T2I is a model that generates images base on text condition, supporting various text encoders and models of different parameter sizes. With minimal training costs, it achieves high-quality image generation by training from scratch. Additionally, it offers usage through CLI console programs and Web Demo displays.

## 📰 News

- [2024-4-1] 🚀🚀🚀 We release the initial version of Lumina-T2I for text-to-image generation

## 🎮 Model Zoo

More checkpoints of our model will be released soon~

| Resolution | Flag-DiT Parameter| Text Encoder | Prediction | Download URL  |
| ---------- | ----------------------- | ------------ | -----------|-------------- |
| 1024       | 5B             |    LLaMa-7B  |   Rectified Flow | [hugging face](https://huggingface.co/Alpha-VLLM/Lumina-T2X/tree/main/Lumina-T2I/5B/1024px) |

Using git for cloning the model you want to use:

```bash
git clone https://huggingface.co/Alpha-VLLM/Lumina-T2X
```

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
  cd Lumina-T2I
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

### CLI

#### Usage

1. Install Lumina-T2I
```bash
pip install -e .
```

2. Run with CLI
```bash
lumina infer "a snow man of ..."
```


#### Inference config structure

```yaml
- settings:

  model:
    ckpt: ""
    ckpt_lm: ""
    token: ""

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
  - `ckpt`: 
  - `ckpt_lm`: 
  - `token`: 
- transport: 
  - `path_type`:   
  - `prediction`:
  - `loss_weight`:
  - `sample_eps`:
  - `train_eps`:
- ode
  - `atol`: 
  - `rtol`: 
  - `reverse`: 
  - `likelihood`: 
- sde
  - `sampling-method`:
  - `diffusion-form`:
  - `diffusion-norm`:
  - `last-step`: 
  - `last-step-size`: 
- infer
  - `resolution`: 
  - `num_sampling_steps`: 
  - `cfg_scale`:   
  - `solver`: 
  - `t_shift`:
  - `ntk_scaling`:     
  - `proportional_attn`:
  - `seed`:              

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
