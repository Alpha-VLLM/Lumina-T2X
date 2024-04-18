<p align="center">
 <img src="../resources/lumina_Intro.png" width="90%"/> 
 <br>
</p>


## Installation
### 1. Create a conda environment and install PyTorch

Note: You may want to adjust the CUDA version [according to your driver version](https://docs.nvidia.com/deploy/cuda-compatibility/#default-to-minor-version).

  ```bash
  conda create -n Lumina_T2X
  conda activate large_dit
  conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
  ```

### 2. Install other dependencies

  ```bash
  pip install diffusers fairscale accelerate tensorboard transformers gradio
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


Before installation, ensure that
you have a working ``nvcc``

```bash
nvcc --version
# The command should work and show the same version number as in step 1 (12.1 in our case).
```

On some outdated distros (e.g., CentOS 7), you may also want to check that a late enough version of
``gcc`` is available

```bash
gcc --version
# The command should work and show a version of at least 6.0.
# If not, consult distro-specific tutorials to obtain a newer version or build manually.
```

Then, clone the repo and install following the official guidelines (note that we expect a full
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


## Model Zoo

The checkpoints of our model will be released soon~

| Resolution | Download URL     |
| ---------- |------------------|
| 256        | To be released   |
| 512        | To be released   |
| 1024       | [hugging face](https://huggingface.co/Alpha-VLLM/Lumina-T2X/tree/main/Lumina-T2I/5B/1024px) |


# Host Local Demo
To host a local gradio demo for interactive inference, run the following command:

```bash
# `/path/to/ckpt` should be a directory containing `consolidated*.pth` and `model_args.pth`

 python -u demo.py --ckpt "/path/to/ckpt"
 
 # use ema model
 python -u demo.py --ckpt "/path/to/ckpt" --ema
```
