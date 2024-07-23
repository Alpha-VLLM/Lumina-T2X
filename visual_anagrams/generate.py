import argparse
from pathlib import Path
from PIL import Image
import json
import math
import os
import random
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from transformers import AutoModel, AutoTokenizer

from visual_anagrams.views import get_anagrams_views
from visual_anagrams.utils import add_args, save_illusion

import models

NEGATIVE_CAP = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic, bad quality, bad detail, blurry-image, jpeg artifacts, bad contrast, bad anatomy, watermark, extra detail, chaotic distribution of objects, distortion"


def get_views(height, width, window_size=128, stride=64):
    # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
    # if panorama's height/width < window_size, num_blocks of height/width should return 1
    num_blocks_height = (
        int((height - window_size) / stride - 1e-6) + 2 if height > window_size else 1
    )
    num_blocks_width = (
        int((width - window_size) / stride - 1e-6) + 2 if width > window_size else 1
    )
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size

        if h_end > height:
            h_start = int(h_start + height - h_end)
            h_end = int(height)
        if w_end > width:
            w_start = int(w_start + width - w_end)
            w_end = int(width)
        if h_start < 0:
            h_end = int(h_end - h_start)
            h_start = 0
        if w_start < 0:
            w_end = int(w_end - w_start)
            w_start = 0

        views.append((h_start, h_end, w_start, w_end))
    return views


def tiled_encode(vae, latents, height, width, sample_size):
    sample_size = 1024
    scale_factor = 8
    core_size = sample_size // 4
    core_stride = core_size
    pad_size = sample_size // 8 * 3
    view_batch_size = 4

    views = get_views(height, width, stride=core_stride, window_size=core_size)
    views_batch = [
        views[i : i + view_batch_size] for i in range(0, len(views), view_batch_size)
    ]
    latents_ = F.pad(latents, (pad_size, pad_size, pad_size, pad_size), "constant", 0)
    image = (
        torch.zeros(latents.size(0), 4, height // scale_factor, width // scale_factor)
        .to(latents.device)
        .to(torch.bfloat16)
    )
    count = torch.zeros_like(image).to(latents.device)
    # get the latents corresponding to the current view coordinates
    for j, batch_view in enumerate(views_batch):
        vb_size = len(batch_view)
        latents_for_view = torch.cat(
            [
                latents_[
                    :, :, h_start : h_end + pad_size * 2, w_start : w_end + pad_size * 2
                ]
                for h_start, h_end, w_start, w_end in batch_view
            ]
        ).to(vae.device)
        image_patch = (
            vae.encode(latents_for_view).latent_dist.sample()
            * vae.config.scaling_factor
        )
        for image_patch_, (h_start, h_end, w_start, w_end) in zip(
            image_patch.chunk(view_batch_size), batch_view
        ):
            h_start, h_end, w_start, w_end = (
                h_start // scale_factor,
                h_end // scale_factor,
                w_start // scale_factor,
                w_end // scale_factor,
            )
            p_h_start, p_h_end, p_w_start, p_w_end = (
                pad_size // scale_factor,
                image_patch.size(2) - pad_size // scale_factor,
                pad_size // scale_factor,
                image_patch.size(3) - pad_size // scale_factor,
            )
            image[:, :, h_start:h_end, w_start:w_end] += image_patch_[
                :, :, p_h_start:p_h_end, p_w_start:p_w_end
            ].to(latents.device)
            count[:, :, h_start:h_end, w_start:w_end] += 1
    image = image / count

    return image


def tiled_decode(vae, latents, height, width, sample_size):
    sample_size = 128
    scale_factor = 8
    core_size = sample_size // 4
    core_stride = core_size
    pad_size = sample_size // 8 * 3
    view_batch_size = 4

    views = get_views(
        height // scale_factor,
        width // scale_factor,
        stride=core_stride,
        window_size=core_size,
    )
    views_batch = [
        views[i : i + view_batch_size] for i in range(0, len(views), view_batch_size)
    ]
    latents_ = F.pad(latents, (pad_size, pad_size, pad_size, pad_size), "constant", 0)
    image = torch.zeros(latents.size(0), 3, height, width).to(latents.device)
    count = torch.zeros_like(image).to(latents.device)
    # get the latents corresponding to the current view coordinates
    for j, batch_view in enumerate(views_batch):
        vb_size = len(batch_view)
        latents_for_view = torch.cat(
            [
                latents_[
                    :, :, h_start : h_end + pad_size * 2, w_start : w_end + pad_size * 2
                ]
                for h_start, h_end, w_start, w_end in batch_view
            ]
        ).to(vae.device)
        image_patch = vae.decode(
            latents_for_view / vae.config.scaling_factor, return_dict=False
        )[0]
        for image_patch_, (h_start, h_end, w_start, w_end) in zip(
            image_patch.chunk(view_batch_size), batch_view
        ):
            h_start, h_end, w_start, w_end = (
                h_start * scale_factor,
                h_end * scale_factor,
                w_start * scale_factor,
                w_end * scale_factor,
            )
            p_h_start, p_h_end, p_w_start, p_w_end = (
                pad_size * scale_factor,
                image_patch.size(2) - pad_size * scale_factor,
                pad_size * scale_factor,
                image_patch.size(3) - pad_size * scale_factor,
            )
            image[:, :, h_start:h_end, w_start:w_end] += image_patch_[
                :, :, p_h_start:p_h_end, p_w_start:p_w_end
            ].to(latents.device)
            count[:, :, h_start:h_end, w_start:w_end] += 1
    image = image / count

    return image


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True
):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


def midpoint_solver(func, t0: float, t1: float, y0: torch.Tensor):
    dt = t1 - t0
    half_dt = 0.5 * dt
    t0_tensor = torch.full((2,), t0).to("cuda")
    f0 = func(y0, t0_tensor)
    y_mid = y0 + f0 * half_dt
    t_mid_tensor = torch.full((2,), t0 + half_dt).to("cuda")
    return func(y_mid, t_mid_tensor) * dt


def midpoint_solver_extra(
    func,
    t0: float,
    t1: float,
    y0: torch.Tensor,
    guidance: torch.Tensor,
    noise: torch.Tensor,
    anchor: torch.Tensor,
    view_fn,
):
    dt = t1 - t0
    half_dt = 0.5 * dt
    t_mid = t0 + half_dt

    # t0
    t0_tensor = torch.full((2,), t0).to("cuda")
    # Skip residual
    c = 0.5 * (1 + torch.cos(torch.pi * torch.tensor(t0))).cpu()  # decay factor
    guidance_t = (t0 * guidance + (1 - t0) * noise) / anchor
    model_input = (1 - c) * y0 + c * guidance_t
    model_input = view_fn.view(model_input[0])
    model_input = torch.stack([model_input] * 2)
    f0 = func(model_input, t0_tensor)

    # Inverse noise
    noise_pred = -f0 * half_dt
    noise_pred = view_fn.inverse_view(noise_pred[0])
    noise_pred = torch.stack([noise_pred] * 2)
    y_mid = y0 - noise_pred

    # t_mid
    t_mid_tensor = torch.full((2,), t_mid).to("cuda")
    # Skip residual
    c = 0.5 * (1 + torch.cos(torch.pi * torch.tensor(t_mid))).cpu()  # decay factor
    guidance_t = (t_mid * guidance + (1 - t_mid) * noise) / anchor
    model_input = (1 - c) * y_mid + c * guidance_t
    model_input = view_fn.view(model_input[0])
    model_input = torch.stack([model_input] * 2)
    f0 = func(model_input, t_mid_tensor)

    return f0 * dt


torch.set_grad_enabled(False)

# Parse args
parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

# Load reference image (for inverse problems)
if args.ref_im_path is not None:
    ref_im = Image.open(args.ref_im_path)
    ref_im = TF.to_tensor(ref_im) * 2 - 1
else:
    ref_im = None

train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

# Load text encoder
print("Creating lm: gemma-2B")
dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
    args.precision
]
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
tokenizer.padding_side = "right"

text_encoder = (
    AutoModel.from_pretrained(
        # gemma_path,
        "google/gemma-2b",
        torch_dtype=dtype,
    )
    .eval()
    .cuda()
)
cap_feat_dim = text_encoder.config.hidden_size

# Load VAE
print(f"Creating vae: {train_args.vae}")
vae = AutoencoderKL.from_pretrained(
    (
        f"stabilityai/sd-vae-ft-{train_args.vae}"
        if train_args.vae != "sdxl"
        else "stabilityai/sdxl-vae"
    ),
    torch_dtype=torch.float32,
).cuda()

# Load DiT
print(f"Creating DiT: {train_args.model}")
model = models.__dict__[train_args.model](
    qk_norm=train_args.qk_norm,
    cap_feat_dim=cap_feat_dim,
    use_flash_attn=args.use_flash_attn,
)
model.eval().to("cuda", dtype=dtype)

# Load DiT checkpoint
if not args.debug:
    if args.ema:
        print("Loading ema model.")
    ckpt = torch.load(
        os.path.join(
            args.ckpt,
            f"consolidated{'_ema' if args.ema else ''}.00-of-01.pth",
        ),
        map_location="cpu",
    )
    model.load_state_dict(ckpt, strict=True)

# Set model args
model_kwargs = {}
model_kwargs["cfg_scale"] = args.cfg_scale

save_dir = Path(args.save_dir) / args.name
save_dir.mkdir(exist_ok=True, parents=True)

# Get prompts embed
cap_feats_list = []
cap_mask_list = []
for p in args.prompts:
    p = f"{args.style} {p}".strip()  # style + description
    cap_feats, cap_mask = encode_prompt(
        [p] + [NEGATIVE_CAP], text_encoder, tokenizer, 0.0
    )
    cap_mask = cap_mask.to(cap_feats.device)
    cap_feats_list.append(cap_feats)
    cap_mask_list.append(cap_mask)

# Get views
views = get_anagrams_views(args.views, view_args=args.view_args)
assert len(args.prompts) == len(views), "Number of prompts must match number of views"

with torch.autocast("cuda", dtype):
    # Sample illusions
    for i in range(args.num_samples):
        # Admin stuff
        torch.random.manual_seed(args.seed + i)
        sample_dir = save_dir / f"{args.seed + i:04}"
        sample_dir.mkdir(exist_ok=True, parents=True)

        for res in args.resolution:
            res_cat, resolution = res.split(":")
            res_cat = int(res_cat)
            scale_factor = res_cat / train_args.image_size
            do_extrapolation = (res_cat // 1) > train_args.image_size

            w, h = resolution.split("x")  # resolution = width * height
            w, h = int(w), int(h)
            w_train, h_train = int(w / scale_factor), int(h / scale_factor)
            latent_w, latent_h = w_train // 8, h_train // 8
            z = torch.randn([1, 4, latent_w, latent_h], device="cuda").to(dtype)
            noisy_img = z.repeat(2, 1, 1, 1)

            timesteps = torch.linspace(0.0, 1.0, args.num_inference_steps)
            if args.time_shifting_factor:
                timesteps = timesteps / (
                    timesteps
                    + args.time_shifting_factor
                    - args.time_shifting_factor * timesteps
                )
            timesteps = timesteps.tolist()

            ################################ Phase Init ##############################
            print("Phase Init...")
            for i, t in enumerate(tqdm(timesteps[:-1])):
                inverted_noises = []
                for j, view_fn in enumerate(views):
                    # Apply views to noisy_img
                    viewed_noisy_img = view_fn.view(noisy_img[0])  # 4 * H * W
                    viewed_noisy_img = torch.stack([viewed_noisy_img] * 2)

                    # Set prompts
                    model_kwargs["cap_feats"] = cap_feats_list[j]
                    model_kwargs["cap_mask"] = cap_mask_list[j]

                    # Predict noise
                    func = partial(model.forward_with_cfg, **model_kwargs)
                    noise = -midpoint_solver(
                        func, timesteps[i], timesteps[i + 1], viewed_noisy_img
                    )

                    # Invert noise
                    inverted_noises.append(view_fn.inverse_view(noise[0]))

                # Reduce noises
                inverted_noises = torch.stack(inverted_noises)
                noise_reduced = inverted_noises.mean(dim=0)

                # Compute the previous noisy img
                noisy_img = noisy_img - noise_reduced

            guidance = vae.decode(noisy_img[:1] / 0.13025).sample
            print(f"Saving illusion preview to {sample_dir}")
            save_illusion(guidance, views, sample_dir)

            ################################ Re-setting ##############################
            if args.proportional_attn:
                model_kwargs["proportional_attn"] = True
                model_kwargs["base_seqlen"] = (train_args.image_size // 16) ** 2
            else:
                model_kwargs["proportional_attn"] = False
                model_kwargs["base_seqlen"] = None

            if do_extrapolation and args.scaling_method == "Time-aware":
                model_kwargs["scale_factor"] = math.sqrt(
                    w * h / train_args.image_size**2
                )
                model_kwargs["scale_watershed"] = args.scaling_watershed
            else:
                model_kwargs["scale_factor"] = 1.0
                model_kwargs["scale_watershed"] = 1.0

            ################################ Phase Upscale ##############################
            # if not do_extrapolation:
            #     continue

            print("Phase Upscale...")
            guidance = guidance.to(torch.float32)
            guidance = F.interpolate(
                guidance, size=(h, w), mode="bicubic"
            )  # upsampling
            guidance = guidance.to(torch.bfloat16)

            print(f"Saving upsampled illusion preview to {sample_dir}")
            save_illusion(guidance, views, sample_dir)

            guidance = tiled_encode(vae, guidance, h, w, train_args.image_size)
            z = (
                torch.randn_like(guidance[:1], device="cuda")
                .to(dtype)
                .repeat(2, 1, 1, 1)
            )
            anchor = (
                torch.ones_like(guidance[:1], device="cuda")
                .to(dtype)
                .repeat(2, 1, 1, 1)
            )
            guidance = guidance.repeat(2, 1, 1, 1)
            noisy_img = z.clone()

            for i, t in enumerate(tqdm(timesteps[:-1])):
                t = torch.tensor(t)
                inverted_noises = []
                for j, view_fn in enumerate(views):
                    # Set prompts
                    model_kwargs["cap_feats"] = cap_feats_list[j]
                    model_kwargs["cap_mask"] = cap_mask_list[j]

                    # Predict nosie
                    func = partial(model.forward_with_cfg, **model_kwargs)
                    noise = -midpoint_solver_extra(
                        func,
                        timesteps[i],
                        timesteps[i + 1],
                        noisy_img,
                        guidance,
                        z,
                        anchor,
                        view_fn,
                    )

                    # Invert noise
                    inverted_noises.append(view_fn.inverse_view(noise[0]))

                # Reduce noises
                inverted_noises = torch.stack(inverted_noises)
                noise_reduced = inverted_noises.mean(dim=0)

                # Compute the previous noisy img
                noisy_img = noisy_img - noise_reduced

            samples = tiled_decode(vae, noisy_img[:1], h, w, train_args.image_size // 8)
            print(f"Saving illusion to {sample_dir}")
            save_illusion(samples, views, sample_dir)
