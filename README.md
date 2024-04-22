<p align="center">
 <img src="./assets/lumina-logo.png" width="40%"/> 
 <br>
</p>

# $\textbf{Lumina-T2X}$: Transform Text into Any Modality, Res. and Duration via Flow-based Large Diffusion Transformer
[[中文版本]](./README_cn.md)

## Introduction

We introduce the $\textbf{Lumina-T2X}$ family, a series of text-conditioned Diffusion Transformers (DiT) designed to convert noise into images, videos, multi-view images of 3D objects and generate speech based on textual instructions. At the core of $\textbf{Lumina-T2X}$ lies the Flow-based Large Diffusion Transformer (Flag-DiT), which supports **scaling up to 7 billion parameters** and **extending sequence lengths up to 128,000**. Inspired by Sora, $\textbf{Lumina-T2X}$ integrates images, videos, multi-views of 3D objects, and speech spectrograms within a spatial-temporal latent token space. 

$\textbf{Lumina-T2X}$ allows for the generation of outputs in **any resolution, aspect ratio, and duration**, facilitated by learnable `newline` and `newframe` tokens.

Furthermore, training $\textbf{Lumina-T2X}$ is computationally efficient. The largest model, with 5 billion parameters, **requires only 20% of the training time needed** for Pixart-alpha, which has 600 million parameters.

🌟 **Features**:
- Flow-based Large Diffusion Transformer (Flag-DiT): Lumina-T2X employs a Large-DiT architecture inspired by the design of GPT-series, ViT-22B, and LLaMA, which is trained with flow matching. The modifications, such as RoPE, RMSNorm, and KQ-norm, over the original DiT, significantly enhance the training stability and model scalability, supporting up to 7 billion parameters and sequences of 128K tokens. We have rigorously ablated the components over the label-conditioned generation on ImageNet, demonstrating faster training convergence, stable training dynamics and a simplified training/inference pipeline. 
- Any Modalities, Res. and Duration within One Framework : Lumina-T2X tokenizes images, videos, multi-views of 3D objects, and spectrograms into one-dimensional sequences, similarly to how Large Language Models (LLMs) process natural languages. With the introduction of learnable placeholders, such as 'nextline' and 'nextframe' tokens, Lumina-T2X can naturally encode any modality—regardless of resolution, aspect ratios, and even temporal durations—into a unified 1-D token sequence akin to LLMs. It then utilizes Flag-DiT with text conditioning to iteratively transform noise into outputs across any modality, resolution, and duration by explicitly specifying the positions of 'nextline' and 'nextframe' during inference time. Remarkably, this flexibility even enables resolution extrapolation, which allows the generation of resolutions out-of-domain that were unseen during training. Specifically, Lumina-T2I can generate images ranging from $768 \times 768$ to $1792 \times 1792$ pixels, even though it was trained at $1024 \times 1024$ pixels, by simply adding more 'nextline' tokens. This discovery significantly broadens the potential applications of Lumina-T2X.
- Low Training Resources : Our empirical observations indicate that the use of larger models, high-resolution images, and extended training durations remarkably enhances the convergence speed of diffusion transformer. Although increasing the token length leads to longer iteration times due to the quadratic complexity of transformers, it substantially reduces the overall training duration by decreasing the required number of iterations. Moreover, by employing meticulously curated text-image and text-video pairs featuring high aesthetic quality frames and detailed captions, our Lumina-T2X model is able to generate high-resolution images and coherent videos with minimal computational demands. Remarkably, the default Lumina-T2I configuration, equipped with a 5 billion Flag-DiT and a 7 billion LLaMa as text encoder, requires only 20\% of the computational resources needed by Pixelart-$\alpha$, which uses a 600 million DiT backbone and 3 billion T5 as text encoders.

![framework](https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/60d2f248-67b1-43ef-a530-c75530cf26c5)

## 📰 News

- [2024-04-19] 🔥🔥🔥 Demo, project introduction and **release**.
- [2024-04-05] 😆😆😆 Code release.
- [2024-04-01] 🚀🚀🚀 We release the initial version of Lumina-T2I for text-to-image generation.

## 📽️ Demos

### Image Generation

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/27bd36a8-8411-47dd-a3a7-3607c1d5d644" width="90%"/> 
 <br>
</p>

### Video Generation

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/2dddcd0c-ce04-4a53-bf5e-060155902ce8" width="90%"/> 
 <br>
</p>

### Multi-view Generation

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/120e5bf9-a7f0-4139-8c53-c9650740f3f7" width="90%"/> 
 <br>
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/cf06c3dc-7102-4548-8955-e3cb6fca1284" width="90%"/> 
</p>

### High-res. Image Editing

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
</p>

## ⚙️ Diverse Configurations

We support diverse configurations, including text encoders, DiTs of different parameter sizes, inference methods, and VAE encoders. Additionally, we offer features such as 1D-RoPE, image enhancement, and more.

<p align="center">
  <img src="./assets/diverse_config.png" width="100%"/>
 <br>
</p>

## 🚀 Quick Start

For more about training and inference, please refer to [Lumina-T2I README.md](./Lumina-T2I/README.md#Installation)

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
