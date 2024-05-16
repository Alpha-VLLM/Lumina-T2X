<!-- <p align="center">
 <img src="./assets/lumina-logo.png" width="40%"/> 
 <br>
</p> -->

# $\textbf{Lumina-T2X}$: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers

<div align="center">

[![GitHub repo contributors](https://img.shields.io/github/contributors-anon/Alpha-VLLM/Lumina-T2X?style=flat&label=Contributors)](https://github.com/Alpha-VLLM/Lumina-T2X/graphs/contributors)
[![GitHub Commit](https://img.shields.io/github/commit-activity/m/Alpha-VLLM/Lumina-T2X?label=Commit)](https://github.com/Alpha-VLLM/Lumina-T2X/commits/main/)
[![Pr](https://img.shields.io/github/issues-pr-closed-raw/Alpha-VLLM/Lumina-T2X.svg?label=Merged+PRs&color=green)](https://github.com/Alpha-VLLM/Lumina-T2X/pulls) <br>
[![GitHub repo stars](https://img.shields.io/github/stars/Alpha-VLLM/Lumina-T2X?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/Alpha-VLLM/Lumina-T2X/stargazers)&#160;
[![GitHub repo watchers](https://img.shields.io/github/watchers/Alpha-VLLM/Lumina-T2X?style=flat&logo=github&logoColor=whitesmoke&label=Watchers)](https://github.com/Alpha-VLLM/Lumina-T2X/watchers)&#160;
[![GitHub repo size](https://img.shields.io/github/repo-size/Alpha-VLLM/Lumina-T2X?style=flat&logo=github&logoColor=whitesmoke&label=Repo%20Size)](https://github.com/Alpha-VLLM/Lumina-T2X/archive/refs/heads/main.zip)&#160;
![Static Badge](https://img.shields.io/badge/-MIT-MIT?logoColor=%231082c3&label=Code%20License&link=https%3A%2F%2Fgithub.com%2FAlpha-VLLM%2FLumina-T2X%2Fblob%2Fmain%2FLICENSE) <br>
[![arXiv](https://img.shields.io/badge/Arxiv-2405.05945-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2405.05945)&#160;
[![Badge](https://img.shields.io/badge/-WeChat@Group-000000?logo=wechat&logoColor=07C160)](http://imagebind-llm.opengvlab.com/qrcode/)&#160;
[![weixin](https://img.shields.io/badge/-WeChat@机器之心-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/NwwbaeRujh-02V6LRs5zMg)&#160;
[![zhihu](https://img.shields.io/badge/-知乎推文-000000?logo=zhihu&logoColor=0084FF)](https://www.zhihu.com/org/opengvlab)&#160;
[![zhihu](https://img.shields.io/badge/-Twitter@OpenGVLab-black?logo=twitter&logoColor=1D9BF0)](https://twitter.com/opengvlab/status/1788949243383910804)&#160;
[![Static Badge](https://img.shields.io/badge/Video%20Introduction%20of%20Lumina--T2X-red?logo=youtube)](https://www.youtube.com/watch?v=KFtHmS5eUCM) <br>
[![Static Badge](https://img.shields.io/badge/Official(node1)-6B88E3?logo=youtubegaming&label=Demo%20Lumina-Next-T2I)](http://106.14.2.150:10020/)&#160;
[![Static Badge](https://img.shields.io/badge/Official(node2)-6B88E3?logo=youtubegaming&label=Demo%20Lumina-Next-T2I)](http://106.14.2.150:10021/)&#160;
[![Static Badge](https://img.shields.io/badge/Official(node3)-6B88E3?logo=youtubegaming&label=Demo%20Lumina-Next-T2I)](http://106.14.2.150:10022/)&#160;
[![Static Badge](https://img.shields.io/badge/Lumina--Next--T2I-HF_Space-yellow?logoColor=violet&label=%F0%9F%A4%97%20Demo%20Lumina-Next-T2I)](https://huggingface.co/spaces/Alpha-VLLM/Lumina-Next-T2I) <br>
[![Static Badge](https://img.shields.io/badge/Lumina--Next--T2I%20checkpoints-Model(2B)-yellow?logoColor=violet&label=%F0%9F%A4%97%20Lumina-Next-T2I%20checkpoints)](https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I)
[![Static Badge](https://img.shields.io/badge/Lumina--T2I%20checkpoints-Model(5B)-yellow?logoColor=violet&label=%F0%9F%A4%97%20Lumina-T2I%20checkpoints)](https://huggingface.co/Alpha-VLLM/Lumina-T2I)

<!-- [![GitHub issues](https://img.shields.io/github/issues/Alpha-VLLM/Lumina-T2X?color=critical&label=Issues)]() -->

<!-- [![GitHub closed issues](https://img.shields.io/github/issues-closed/Alpha-VLLM/Lumina-T2X?color=success&label=Issues)]() <br> -->

<!-- [![GitHub repo forks](https://img.shields.io/github/forks/Alpha-VLLM/Lumina-T2X?style=flat&logo=github&logoColor=whitesmoke&label=Forks)](https://github.com/Alpha-VLLM/Lumina-T2X/network)  -->

[[📄 Lumina-T2X arXiv](https://arxiv.org/abs/2405.05945)]
[[📽️ Video Introduction of Lumina-T2X](https://www.youtube.com/watch?v=KFtHmS5eUCM)]
[👋 join our <a href="http://imagebind-llm.opengvlab.com/qrcode/" target="_blank">WeChat</a>]

[[🤖️ Lumina-T2I 5B Checkpoints](https://huggingface.co/Alpha-VLLM/Lumina-T2I)]
[[🤖️ Lumina-Next-T2I 2B Checkpoints (recommend)](https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I)]

[[🔮 GUI Demo for Lumina-Next-T2I 2B model (node1)](http://106.14.2.150:10020/)]

[[🔮 GUI Demo for Lumina-Next-T2I 2B model (node2)](http://106.14.2.150:10021/)]

[[🔮 GUI Demo for Lumina-Next-T2I 2B model (node3)](http://106.14.2.150:10022/)]




<!-- [[📺 Website](https://lumina-t2-x-web.vercel.app/)] -->

</div>

![intro_large](https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/9f52eabb-07dc-4881-8257-6d8a5f2a0a5a)

<!-- [[中文版本]](./README_cn.md) -->

## 📰 News
- [2024-05-14] 🔥🔥🔥 Lumina-Next now supports simple **text-to-music** generation ([examples](#text-to-music-generation)), **high-resolution (1024*4096) Panorama** generation conditioned on text ([examples](#panorama-generation)), and **3D point cloud** generation conditioned on labels ([examples](#point-cloud-generation)).
- [2024-05-13] 🔥🔥🔥 We give [examples](#multilingual-generation) demonstrating Lumina-T2X's capability to support **multilingual prompts**, and even support prompts containing **emojis**.
- **[2024-05-12]** 🤩🤩🤩 **We excitedly released our `Lumina-Next-T2I` model ([checkpoint](https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I)) which uses a 2B Next-DiT model as the backbone and Gemma-2B as the text encoder. Try it out at [demo1](http://106.14.2.150:10020/) & [demo2](http://106.14.2.150:10021/) & [demo3](http://106.14.2.150:10022/).**
- **[2024-05-10]** 🔥🔥🔥 **We released the technical report on [arXiv](https://arxiv.org/abs/2405.05945).**
- **[2024-05-09]** 🚀🚀🚀 **We released `Lumina-T2A` (Text-to-Audio) Demos. [Examples](#text-to-audio-generation)**
- [2024-04-29] 🔥 We released the 5B model [checkpoint](https://huggingface.co/Alpha-VLLM/Lumina-T2I) and demo built upon it for text-to-image generation.
- [2024-04-25] 🔥 Support 720P video generation with arbitrary aspect ratio. [Examples](#text-to-video-generation) 🚀🚀🚀
- [2024-04-19]  Demo examples released.
- [2024-04-05] Code released for `Lumina-T2I`.
- [2024-04-01] We release the initial version of `Lumina-T2I` for text-to-image generation.

## 🚀 Quick Start

> [!Warning]
> **Since we are updating the code frequently, please pull the latest code:**
> ```bash
> git pull origin main
> ```

In order to quickly get you guys using our model, we built different versions of the GUI demo site.

#### Lumina-Next-T2I 2B model demo:
[[node1](http://106.14.2.150:10020/)] [[node2](http://106.14.2.150:10021/)] [[node3](http://106.14.2.150:10022/)]


For more details about training and inference, please refer to [Lumina-T2I](./lumina_t2i/README.md#Installation) and [Lumina-Next-T2I](./lumina_nextr_t2i/README.md#Installation)

> [!Warning]
> **Lumina-T2X employs FSDP for training large diffusion models. FSDP shards parameters, optimizer states, and gradients across GPUs. Thus, at least 8 GPUs are required for full fine-tuning of the Lumina-T2X 5B model. Parameter-efficient Finetuning of Lumina-T2X shall be released soon.**

Installation on your environment:

```bash
pip install git+https://github.com/Alpha-VLLM/Lumina-T2X
```

## 📑 Open-source Plan

- [X] Lumina-T2I (Demos✅, Training✅, Inference✅, Checkpoints✅)
- [ ] Lumina-T2V (Demos✅)
- [ ] Lumina-T2A (Demos✅)
- [X] Web Demo
- [X] Cli Demo

## 📜 Index of Content

- [Lumina-T2X](#lumina-t2x)
  - [📰 News](#-news)
  - [🚀 Quick Start](#-quick-start)
  - [📑 Open-source Plan](#-open-source-plan)
  - [📜 Index of Content](#-index-of-content)
  - [Introduction](#introduction)
  - [📽️ Demo Examples](#️-demo-examples)
    - [Text-to-Image Generation](#text-to-image-generation)
    - [Text-to-Video Generation](#text-to-video-generation)
    - [Text-to-3D Generation](#text-to-3d-generation)
    - [Text-to-Audio Generation](#text-to-audio-generation)
    - [Text-to-music Generation](#text-to-music-generation)
    - [Multilingual Examples](#multilingual-generation)
  - [⚙️ Diverse Configurations](#️-diverse-configurations)

## Introduction

We introduce the $\textbf{Lumina-T2X}$ family, a series of text-conditioned Diffusion Transformers (DiT) capable of transforming textual descriptions into vivid images, dynamic videos, detailed multi-view 3D images, and synthesized speech. At the core of Lumina-T2X lies the **Flow-based Large Diffusion Transformer (Flag-DiT)**—a robust engine that supports up to **7 billion parameters** and extends sequence lengths to **128,000** tokens. Drawing inspiration from Sora, Lumina-T2X integrates images, videos, multi-views of 3D objects, and speech spectrograms within a spatial-temporal latent token space, and can generate outputs at **any resolution, aspect ratio, and duration**.

🌟 **Features**:

- **Flow-based Large Diffusion Transformer (Flag-DiT)**: Lumina-T2X adopts the **flow matching** formulation and is equipped with many advanced techniques, such as RoPE, RMSNorm, and KQ-norm, **demonstrating faster training convergence, stable training dynamics, and a simplified pipeline**.
- **Any Modalities, Resolution, and Duration within One Framework**:
  1. $\textbf{Lumina-T2X}$ can **encode any modality, including mages, videos, multi-views of 3D objects, and spectrograms into a unified 1-D token sequence at any resolution, aspect ratio, and temporal duration.**
  2. By introducing the `[nextline]` and `[nextframe]` tokens, our model can **support resolution extrapolation**, i.e., generating images/videos with out-of-domain resolutions **not encountered during training**, such as images from 768x768 to 1792x1792 pixels.
- **Low Training Resources**: Our empirical observations indicate that employing larger models,
  high-resolution images, and longer-duration video clips can **significantly accelerate the convergence**
  **speed** of diffusion transformers. Moreover, by employing meticulously curated text-image and text-video pairs featuring high aesthetic quality frames and detailed captions, our $\textbf{Lumina-T2X}$ model is learned to generate high-resolution images and coherent videos with minimal computational demands. Remarkably, the default Lumina-T2I configuration, equipped with a 5B Flag-DiT and a 7B LLaMA as the text encoder, **requires only 35% of the computational resources compared to Pixelart-**$\alpha$.

![framework](https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/60d2f248-67b1-43ef-a530-c75530cf26c5)

## 📽️ Demo Examples

### Text-to-Image Generation

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/27bd36a8-8411-47dd-a3a7-3607c1d5d644" width="90%"/> 
 <br>
</p>

#### Panorama Generation

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/88b75b4e-5e16-4ea3-aba8-134904dd3381" width="90%"/> 
 <br>
</p>

### Text-to-Video Generation

**720P Videos:**

**Prompt:** The majestic beauty of a waterfall cascading down a cliff into a serene lake.

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/17187de8-7a07-49a8-92f9-fdb8e2f5e64c

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/0a20bb39-f6f7-430f-aaa0-7193a71b256a

**Prompt:** A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/7bf9ce7e-f454-4430-babe-b14264e0f194

**360P Videos:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/d7fec32c-3655-4fd1-aa14-c0cb3ace3845

### Text-to-3D Generation

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/cd061b8d-c47b-4c0c-b775-2cbaf8014be9

#### Point Cloud Generation

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/742237ad-be47-4a7d-aa11-b3aaba07a75a" width="90%"/> 
 <br>
</p>

### Text-to-Audio Generation

> [!Note]
> **Attention: Mouse over the playbar and click the audio button on the playbar to unmute it.**

<!-- > 🌟🌟🌟 **We recommend visiting the Lumina website to try it out! [🌟 visit](https://lumina-t2-x-web.vercel.app/docs/demos/demo-of-audio)** -->

**Prompt:** Semiautomatic gunfire occurs with slight echo

**Generated Audio:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/25f2a6a8-0386-41e8-ab10-d1303554b944

**Groundtruth:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/6722a68a-1a5a-4a44-ba9c-405372dc27ef

**Prompt:** A telephone bell rings

**Generated Audio:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/7467dd6d-b163-4436-ac5b-36662d1f9ddf

**Groundtruth:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/703ea405-6eb4-4161-b5ff-51a93f81d013

**Prompt:** An engine running followed by the engine revving and tires screeching

**Generated Audio:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/5d9dd431-b8b4-41a0-9e78-bb0a234a30b9

**Groundtruth:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/9ca4af9e-cee3-4596-b826-d6c25761c3c1

**Prompt:** Birds chirping with insects buzzing and outdoor ambiance

**Generated Audio:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/b776aacb-783b-4f47-bf74-89671a17d38d

**Groundtruth:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/a11333e4-695e-4a8c-8ea1-ee5b83e34682

### Text-to-music Generation

**Prompt:** An electrifying ska tune with prominent saxophone riffs, energetic e-guitar and acoustic drums, lively percussion, soulful keys, groovy e-bass, and a fast tempo that exudes uplifting energy.

**Generated Music:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/fef8f6b9-1e77-457e-bf4b-fb0cccefa0ec

**Prompt:** A high-energy synth rock/pop song with fast-paced acoustic drums, a triumphant brass/string section, and a thrilling synth lead sound that creates an adventurous atmosphere.

**Generated Music:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/1f796046-64ab-44ed-a4d8-0ebc0cfc484f

**Prompt:** An uptempo electronic pop song that incorporates digital drums, digital bass and synthpad sounds.

**Generated Music:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/4768415e-436a-4d0e-af53-bf7882cb94cd

**Prompt:** A medium-tempo digital keyboard song with a jazzy backing track featuring digital drums, piano, e-bass, trumpet, and acoustic guitar.

**Generated Music:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/8994a573-e776-488b-a86c-4398a4362398

**Prompt:** This low-quality folk song features groovy wooden percussion, bass, piano, and flute melodies, as well as sustained strings and shimmering shakers that create a passionate, happy, and joyful atmosphere.

**Generated Music:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/e0b5d197-589c-47d6-954b-b9c1d54feebb

### Multilingual Generation

We present three multilingual capabilities of Lumina-Next-2B.

**Generating Images conditioned on Chinese poems:**

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/9aa79d67-e304-4867-81f3-cfc934c625d9" width="90%"/> 
 <br>
</p>

**Generating Images with multilignual prompts:**


<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/7c62bb94-42e4-4525-a298-9e25475b511d" width="90%"/> 
 <br>
</p>

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/07fc8138-e67c-4c9f-bc01-e749a6507ada" width="90%"/> 
 <br>
</p>

**Generating Images with emojis:**

<p align="center">
 <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/86041420/980b4999-9d1c-4fbd-a695-88b6b675f34b" width="90%"/> 
 <br>
</p>

<!--
**Prompt:** Water trickling rapidly and draining

**Generated Audio:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/88fcf0e1-b71a-4e94-b9a6-138db6a670f0

**Groundtruth:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/6fb9963f-46a5-4020-b160-f9a004528d7e

**Prompt:** Thunderstorm sounds while raining

**Generated Audio:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/fad8baf3-d80b-4915-ba31-aab13db5ce06

**Groundtruth:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/c01a7e6e-3421-4a28-93c5-831523ec061d

**Prompt:** Birds chirping repeatedly

**Generated Audio:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/0fa673a3-f9de-487b-8812-1f96a335e913

**Groundtruth:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/718289f9-a93e-4ea9-b7db-a14c2b209b28

**Prompt:** Several large bells ring

**Generated Audio:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/362fde84-e4ae-4152-aeb5-4355155c8719

**Groundtruth:**

https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/da93e13d-6462-48d2-b6dc-af6ff0c4d07d

-->

<!-- For more audio demos visit [lumina website - audio demos](https://lumina-t2-x-web.vercel.app/docs/demos/demo-of-audio) -->

<!-- ### More examples -->

<!-- For more demos visit [this website](https://lumina-t2-x-web.vercel.app/docs/demos) -->

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
  <img src="https://github.com/Alpha-VLLM/Lumina-T2X/assets/54879512/221de325-d9fb-4b7e-a97c-4b24cd2df0fc" width="100%"/>
 <br>
</p>

## 📄 Citation

```
@article{gao2024luminat2x,
      title={Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers}, 
      author={Peng Gao and Le Zhuo and Ziyi Lin and Chris Liu and Junsong Chen and Ruoyi Du and Enze Xie and Xu Luo and Longtian Qiu and Yuhang Zhang and Chen Lin and Rongjie Huang and Shijie Geng and Renrui Zhang and Junlin Xi and Wenqi Shao and Zhengkai Jiang and Tianshuo Yang and Weicai Ye and He Tong and Jingwen He and Yu Qiao and Hongsheng Li},
      journal={arXiv preprint arXiv:2405.05945},
      year={2024}
}
```

<!-- 
## Star History

 [![Star History Chart](https://api.star-history.com/svg?repos=Alpha-VLLM/Lumina-T2X&type=Date)](https://star-history.com/#Alpha-VLLM/Lumina-T2X&Date) -->
