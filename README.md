<!-- <p align="center">
 <img src="./assets/lumina-logo.png" width="40%"/> 
 <br>
</p> -->

# $\textbf{Lumina-T2X}$: Transform Text into Any Modality, Res. and Duration via Flow-based Large Diffusion Transformer

[![GitHub repo contributors](https://img.shields.io/github/contributors-anon/Alpha-VLLM/Lumina-T2X?style=flat&label=Contributors)](https://github.com/Alpha-VLLM/Lumina-T2X/graphs/contributors) 
[![GitHub Commit](https://img.shields.io/github/commit-activity/m/Alpha-VLLM/Lumina-T2X?label=Commit)](https://github.com/Alpha-VLLM/Lumina-T2X/commits/main/)
[![Pr](https://img.shields.io/github/issues-pr-closed-raw/Alpha-VLLM/Lumina-T2X.svg?label=Merged+PRs&color=green)](https://github.com/Alpha-VLLM/Lumina-T2X/pulls)
[![GitHub repo stars](https://img.shields.io/github/stars/Alpha-VLLM/Lumina-T2X?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/Alpha-VLLM/Lumina-T2X/stargazers)&#160;
[![GitHub repo watchers](https://img.shields.io/github/watchers/Alpha-VLLM/Lumina-T2X?style=flat&logo=github&logoColor=whitesmoke&label=Watchers)](https://github.com/Alpha-VLLM/Lumina-T2X/watchers)&#160;
[![GitHub repo size](https://img.shields.io/github/repo-size/Alpha-VLLM/Lumina-T2X?style=flat&logo=github&logoColor=whitesmoke&label=Repo%20Size)](https://github.com/Alpha-VLLM/Lumina-T2X/archive/refs/heads/main.zip)
<!-- [![GitHub issues](https://img.shields.io/github/issues/Alpha-VLLM/Lumina-T2X?color=critical&label=Issues)](https://github.com/PKU-YuanGroup/Video-LLaVA/issues?q=is%3Aopen+is%3Aissue) -->
<!-- [![GitHub closed issues](https://img.shields.io/github/issues-closed/Alpha-VLLM/Lumina-T2X?color=success&label=Issues)](https://github.com/PKU-YuanGroup/Video-LLaVA/issues?q=is%3Aissue+is%3Aclosed) <br> -->
<!-- [![GitHub repo forks](https://img.shields.io/github/forks/Alpha-VLLM/Lumina-T2X?style=flat&logo=github&logoColor=whitesmoke&label=Forks)](https://github.com/Alpha-VLLM/Lumina-T2X/network)&#160; -->


![intro](./assets/intro_large_update.png)

<!-- [[中文版本]](./README_cn.md) -->

## 📰 News

- **[2024-04-25]** 🔥🔥🔥 **Support 720p video generation with arbitary resolution. [Demo](#video-generation)** 🚀🚀🚀
- [2024-04-19] 🔥🔥🔥 Demo released.
- [2024-04-05] 😆😆😆 Code released for Lumina-T2I.
- [2024-04-01] 🚀🚀🚀 We release the initial version of Lumina-T2I for text-to-image generation.

## 🚀 Quick Start

For more about training and inference, please refer to [Lumina-T2I README.md](./Lumina-T2I/README.md#Installation)

## 📑 Open-source Plan

- [x] Lumina-T2I (Training, Inference)
- [ ] Lumina-T2V
- [x] Web Demo
- [x] Cli Demo

## 📜 Index of Content

- [Lumina-T2X](#textbflumina-t2x-transform-text-into-any-modality-res-and-duration-via-flow-based-large-diffusion-transformer)
  - [📰 News](#-news)
  - [🚀 Quick Start](#-quick-start)
  - [📑 Opensource Plan](#-opensource-plan)
  - [📜 Index of Content](#-index-of-content)
  - [Introduction](#introduction)
  - [📽️ Demos](#️-demos)
    - [Image Generation](#image-generation)
    - [Video Generation](#video-generation)
    - [Multi-view Generation](#multi-view-generation)
    - [More demos](#more-demos)
  - [⚙️ Diverse Configurations](#️-diverse-configurations)

## Introduction

We introduce the $\textbf{Lumina-T2X}$ family, a series of text-conditioned Diffusion Transformers (DiT) designed to convert noise into images, videos, and multi-view images of 3D objects and generate speech based on textual instructions. At the core of Lumina-T2X lies the Flow-based Large Diffusion Transformer (Flag-DiT), which supports **scaling up to 7 billion parameters** and **extending sequence lengths up to 128,000**. Inspired by Sora, Lumina-T2X integrates images, videos, multi-views of 3D objects, and speech spectrograms within a spatial-temporal latent token space. 

$\textbf{Lumina-T2X}$ allows for the generation of outputs in **any resolution, aspect ratio, and duration**, facilitated by learnable `newline` and `newframe` tokens.

Furthermore, training $\textbf{Lumina-T2X}$ is computationally efficient. The largest model, with 5 billion parameters, **requires only 20% of the training time needed** for Pixart-alpha, which has 600 million parameters.

🌟 **Features**:
- Flow-based Large Diffusion Transformer (Flag-DiT): Lumina-T2X is trained **with the flow matching object**. To increase training stability and model scalability, we support a bunch of techniques, such as RoPE, RMSNorm, and KQ-norm, **demonstrating faster training convergence, stable training dynamics, and a simplified pipeline**.
- Any Modalities, Res., and Duration within One Framework: 
  1. Lumina-T2X tokenizes images, videos, multi-views of 3D objects, and spectrograms into one-dimensional sequences. 
  2. Lumina-T2X can naturally **encode any modality—regardless of resolution, aspect ratios, and temporal durations into a unified 1-D token sequence** akin to LLMs, by utilizing Flag-DiT with text conditioning to iteratively transform noise into outputs across any modality, resolution, and duration during inference time. 
  3. Due to the use of `nextline` and `nextframe` tokens, our model can **support resolution extrapolation**, which allows the generation of resolutions out-of-domain that **were unseen during training**.
- Low Training Resources: increasing token length in transformers extends iteration times but **reduces overall training duration by decreasing the number of iterations needed**. Moreover, our Lumina-T2X model can generate high-resolution images and coherent videos **with minimal computational demands**. Remarkably, the default Lumina-T2I configuration, equipped with a 5B Flag-DiT and a 7B LLaMA as text encoder, **requires only 20% of the computational resources needed by Pixelart-**$\alpha$.

![framework](https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/60d2f248-67b1-43ef-a530-c75530cf26c5)

## 📽️ Demos

### Text-to-Image Generation

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/27bd36a8-8411-47dd-a3a7-3607c1d5d644" width="90%"/> 
 <br>
</p>

### Text-to-Video Generation

**720P Videos:**

**Prompt:** The majestic beauty of a waterfall cascading down a cliff into a serene lake.

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/17187de8-7a07-49a8-92f9-fdb8e2f5e64c


**Prompt:** A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/0a20bb39-f6f7-430f-aaa0-7193a71b256a

360P Videos:

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/d7fec32c-3655-4fd1-aa14-c0cb3ace3845

### Text-to-Multi-views Generation

<p align="center">
 <img src="./assets/videos/multi_view_all_fix.gif" width="90%"/> 
</p>


### More demos

For more demos visit [this website](https://lumina-t2-x-web.vercel.app)
<!-- ### High-res. Image Editing

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/55981976-c989-4f07-982a-1e567c7078ef" width="90%"/> 
 <br>
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/a1ac7190-c49c-4d8b-965c-9ccf83a4f6a7" width="90%"/> 
</p>

### Compositional Generation

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/8c8eb921-134c-4f55-918a-0ad07f9a47f4" width="90%"/> 
 <br>
</p>

### Resolution Extrapolation

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/e37e2db7-3ead-451e-ba18-b375eb773578" width="90%"/> 
 <br>
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/9da47c34-5e09-48d3-9c48-78663fd01cc8" width="100%"/> 
</p>

### Consistent-Style Generation

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/6403417a-42c6-4048-9419-375d211e14bb" width="90%"/> 
 <br>
</p> -->

## ⚙️ Diverse Configurations

We support diverse configurations, including text encoders, DiTs of different parameter sizes, inference methods, and VAE encoders. Additionally, we offer features such as 1D-RoPE, image enhancement, and more.

<p align="center">
  <img src="./assets/diverse_config.png" width="100%"/>
 <br>
</p>


<!--

## 📄 Citation

```
@inproceedings{luminat2x,
  author    = {},
  title     = {},
  booktitle = {},
  pages     = {}
  year      = {2024}
}
```

## Star History

 [![Star History Chart](https://api.star-history.com/svg?repos=Alpha-VLLM/Lumina-T2X&type=Date)](https://star-history.com/#Alpha-VLLM/Lumina-T2X&Date) -->
