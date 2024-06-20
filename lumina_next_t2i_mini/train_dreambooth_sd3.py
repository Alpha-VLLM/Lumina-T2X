# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Lumina-T2I using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import partial
import json
import logging
import os
import random
import socket
from time import time

from PIL import Image
from diffusers import AutoencoderKL, SD3Transformer2DModel, StableDiffusion3Pipeline
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from transformers.utils import logging as hf_logging
import yaml

from data import ItemProcessor, MyDataset, read_general
from grad_norm import calculate_l2_grad_norm, scale_grad
from imgproc import var_center_crop
from parallel import distributed_init, get_intra_node_process_group
from transport import training_losses

hf_logging.set_verbosity_error()


#############################################################################
#                            Data item Processor                            #
#############################################################################


class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform):
        self.image_transform = transform

    def process_item(self, data_item, training_mode=False):
        if "conversations" in data_item:
            assert "image" in data_item and len(data_item["conversations"]) == 2
            image = Image.open(read_general(data_item["image"])).convert("RGB")
            text = data_item["conversations"][1]["value"]
        elif "path" in data_item:
            image_path = data_item["path"]
            image = Image.open(read_general(image_path)).convert("RGB")
            text = data_item["prompt"]
        elif "image_url" in data_item:
            url = data_item["image_url"]
            url = url.replace(
                "/mnt/petrelfs/share_data/gaopeng/image_text_data",
                "/mnt/hwfile/alpha_vl/gaopeng/share_data/image_text_data",
            )
            url = url.replace("/mnt/petrelfs/share_data/huxiangfei", "/mnt/hwfile/alpha_vl/huxiangfei")
            image = Image.open(read_general(url)).convert("RGB")
            caption_keys = [
                "sharegpt4v_long_cap",
                "cogvlm_long",
                "blip2_short_cap",
                "llava13b_long_cap",
                "spatial_caption",
                "coca_caption",
                "user_prompt",
                "tags_prompt",
                "gpt4v_concise_elements",
                "gpt4v_regions_detailed_description",
                "gpt4v_detailed_description",
                "gpt4v_concise_description",
            ]
            candidate_caps = [data_item[x] for x in caption_keys if x in data_item and data_item[x]]
            text = random.choice(candidate_caps) if len(candidate_caps) else ""
        elif "image" in data_item and "caption" in data_item:
            image = Image.open(read_general(data_item["image"].replace("hzh:s3://", "cluster_s_hdd_gp:s3://"))).convert(
                "RGB"
            )
            text = data_item["caption"]
        else:
            raise ValueError(f"Unrecognized item: {data_item}")

        image = self.image_transform(image)

        return image, text


def generate_json(data_path, instance_prompt, save_path=".tmp/meta_data.yaml"):

    all_imgs = os.listdir(data_path)
    all_imgs = [os.path.join(data_path, x) for x in all_imgs if x.split(".")[-1] in ["jpg", "jpeg", "png"]]

    data = [{"prompt": instance_prompt, "path": x} for x in all_imgs]

    os.makedirs(".tmp", exist_ok=True)
    with open(".tmp/data.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    meta_data = {
        "META": [
            {
                "path": ".tmp/data.json",
                "type": "image_text",
            },
        ]
    }

    with open(save_path, "w") as yaml_file:
        yaml.dump(meta_data, yaml_file, default_flow_style=False)

    return save_path


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


#############################################################################
#                           Training Helper Functions                       #
#############################################################################


def dataloader_collate_fn(samples):
    image = [x[0] for x in samples]
    caps = [x[1] for x in samples]
    return image, caps


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    if hasattr(model, "text_model"):
        wrap_blocks = list(model.text_model.encoder.layers)
    else:
        wrap_blocks = list(model.encoder.block)
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in wrap_blocks,
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
        forward_prefetch=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.transformer_blocks),
        ),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


#############################################################################
#                                Training Loop                              #
#############################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    mp_world_size = 1
    mp_rank = 0

    assert args.global_batch_size % dp_world_size == 0, "Batch size must be divisible by data parrallel world size."
    local_batch_size = args.global_batch_size // dp_world_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        tb_logger = SummaryWriter(
            os.path.join(
                args.results_dir,
                "tensorboard",
                datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname(),
            )
        )
    else:
        logger = create_logger(None)
        tb_logger = None

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.model_path,
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.model_path,
        subfolder="tokenizer_2",
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.model_path,
        subfolder="tokenizer_3",
    )
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]

    # Load text encoders
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(args.model_path, subfolder="text_encoder")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.model_path, subfolder="text_encoder_2")
    text_encoder_three = T5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder_3")
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    # Load vae
    vae = AutoencoderKL.from_pretrained(
        args.model_path,
        subfolder="vae",
    )

    # Load model
    model = SD3Transformer2DModel.from_pretrained(args.model_path, subfolder="transformer")

    model.requires_grad_(True)
    vae.requires_grad_(False)
    if args.train_text_encoder:
        text_encoder_one.requires_grad_(True)
        text_encoder_two.requires_grad_(True)
        text_encoder_three.requires_grad_(True)
    else:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_three.requires_grad_(False)

    vae = vae.to(device, dtype=torch.float32)
    if not args.train_text_encoder:
        text_encoder_one = text_encoder_one.to(device, dtype=torch.float32)
        text_encoder_two = text_encoder_two.to(device, dtype=torch.float32)
        text_encoder_three = text_encoder_three.to(device, dtype=torch.float32)

    model_ema = deepcopy(model)

    # resume (part 1, before FSDP wrapping)
    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")
    if args.resume:
        if dp_rank == 0:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.resume}")
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
            logger.info(f"Resuming ema weights from: {args.resume}")
            model_ema.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated_ema.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
    elif args.init_from:
        if dp_rank == 0:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(
                os.path.join(
                    args.init_from,
                    f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
                ),
                map_location="cpu",
            )

            size_mismatch_keys = []
            model_state_dict = model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")

    # checkpointing (part1, should be called before FSDP wrapping)
    if args.checkpointing:
        checkpointing_list = list(model.transformer_blocks)
        checkpointing_list_ema = list(model_ema.transformer_blocks)

    # FSDP wrapping
    model = setup_fsdp_sync(model, args)  # todo fsdp_wrap_module_list
    model_ema = setup_fsdp_sync(model_ema, args)

    # checkpointing (part2, after FSDP wrapping)
    # gradient checkpointing
    if args.checkpointing:
        print("apply gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list,
        )
        apply_activation_checkpointing(
            model_ema,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_ema,
        )

    dist.barrier()

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant
    # learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    # Setup data:
    logger.info("Creating data transform...")
    crop_size_list = [(1024, 1024)]  # todo variable resolution support
    logger.info("List of crop sizes:")
    for i in range(0, len(crop_size_list), 6):
        logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[i : i + 6]]))
    image_transform = transforms.Compose(
        [
            transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    data_path = generate_json(args.data_path, args.instance_prompt)
    dataset = MyDataset(
        data_path,
        item_processor=T2IItemProcessor(image_transform),
        cache_on_disk=args.cache_data_on_disk,
    )
    num_samples = args.global_batch_size * args.max_steps
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Total # samples to consume: {num_samples:,} " f"({num_samples / len(dataset):.2f} epochs)")
    sampler = get_train_sampler(
        dataset,
        dp_rank,
        dp_world_size,
        args.global_batch_size,
        args.max_steps,
        resume_step,
        args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataloader_collate_fn,
    )

    # Prepare models for training:
    # important! This enables embedding dropout for classifier-free guidance
    model.train()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_grad_norm = 0
    start_time = time()

    logger.info(f"Training for {args.max_steps:,} steps...")
    for step, (x, caps) in enumerate(loader, start=resume_step):
        # caps: List[str]
        x = [img.to(device, non_blocking=True) for img in x]

        with torch.no_grad():
            vae_scale = {"sdxl": 0.13025, "sd3": 1.5305, "ema": 0.18215, "mse": 0.18215}["sd3"]
            vae_shift = {"sdxl": 0.0, "sd3": 0.0609, "ema": 0.0, "mse": 0.0}["sd3"]

            if step == resume_step:
                logger.warning(f"vae scale: {vae_scale}    vae shift: {vae_shift}")
            # Map input images to latent space + normalize latents:
            x = [(vae.encode(img[None]).latent_dist.sample()[0] - vae_shift) * vae_scale for img in x]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, args.instance_prompt, device="cuda"
            )

        loss_item = 0.0
        opt.zero_grad()
        for mb_idx in range((local_batch_size - 1) // args.micro_batch_size + 1):
            mb_st = mb_idx * args.micro_batch_size
            mb_ed = min((mb_idx + 1) * args.micro_batch_size, local_batch_size)
            last_mb = mb_ed == local_batch_size

            x_mb = torch.stack(x[mb_st:mb_ed], dim=0)
            prompt_embeds_mb = prompt_embeds[mb_st:mb_ed]
            pooled_prompt_embeds_mb = pooled_prompt_embeds[mb_st:mb_ed]

            model_kwargs = dict(
                encoder_hidden_states=prompt_embeds_mb,
                pooled_projections=pooled_prompt_embeds_mb,
            )
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:

                def model_func(latents, t, **kwargs):
                    # SD3 uses a reversed notation for the flow between pure signal and noise
                    # specifically, at timestep 0, the latents are pure noise for lumina-series models,
                    # while for sd3, the latents are pure signal
                    # additional, the timesteps that sd3 model expects lie between [0,1000], while
                    # for lumia the range is [0, 1]
                    # this function wraps the model to eliminate these gaps
                    result = model(hidden_states=latents, timestep=(1 - t) * 1000, **kwargs)[0]
                    return -result

                loss_dict = training_losses(model_func, x_mb, model_kwargs)
            loss = loss_dict["loss"].sum() / local_batch_size
            loss_item += loss.item()
            with model.no_sync() if args.data_parallel in ["sdp", "fsdp"] and not last_mb else contextlib.nullcontext():
                loss.backward()

        grad_norm = calculate_l2_grad_norm(model)
        if grad_norm > args.grad_clip:
            scale_grad(model, args.grad_clip / grad_norm)

        if tb_logger is not None:
            tb_logger.add_scalar("train/loss", loss_item, step)
            tb_logger.add_scalar("train/grad_norm", grad_norm, step)
            tb_logger.add_scalar("train/lr", opt.param_groups[0]["lr"], step)

        opt.step()
        update_ema(model_ema, model)

        # Log loss values:
        running_loss += loss_item
        running_grad_norm += grad_norm
        log_steps += 1
        if (step + 1) % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            secs_per_step = (end_time - start_time) / log_steps
            imgs_per_sec = args.global_batch_size * log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            avg_grad_norm = running_grad_norm / log_steps
            logger.info(
                f"(step={step + 1:07d}) "
                f"Train Loss: {avg_loss:.4f}, "
                f"Train Grad Norm: {avg_grad_norm:.4f}, "
                f"Train Secs/Step: {secs_per_step:.2f}, "
                f"Train Imgs/Sec: {imgs_per_sec:.2f},"
            )
            # Reset monitoring variables:
            running_loss = 0
            running_grad_norm = 0
            log_steps = 0
            start_time = time()

        # Save DiT checkpoint:
        if step == 0 or (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if dp_rank == 0:
                    consolidated_fn = "consolidated." f"{mp_rank:02d}-of-" f"{mp_world_size:02d}" ".pth"
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_ema_state_dict = model_ema.state_dict()
                if dp_rank == 0:
                    consolidated_ema_fn = "consolidated_ema." f"{mp_rank:02d}-of-" f"{mp_world_size:02d}" ".pth"
                    torch.save(
                        consolidated_ema_state_dict,
                        os.path.join(checkpoint_path, consolidated_ema_fn),
                    )
            dist.barrier()
            del consolidated_ema_state_dict
            logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT_Llama2_7B_patch2 with the
    # hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--train_text_encoder", action="store_true", default=False)
    parser.add_argument("--image_size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--checkpointing", action="store_true", default=False, help="enable gradient checkpointing")
    parser.add_argument(
        "--local_diffusers_model_root",
        type=str,
        help="Specify the root directory if diffusers models are to be loaded "
        "from the local filesystem (instead of being automatically "
        "downloaded from the Internet). Useful in environments without "
        "Internet access.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--init_from",
        type=str,
        help="Initialize the model weights from a checkpoint folder. "
        "Compared to --resume, this loads neither the optimizer states "
        "nor the data loader states.",
    )
    parser.add_argument("--use_t5", action="store_true", default=True)
    parser.add_argument("--no_t5", action="store_false", dest="use_t5")
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=2.0,
        help="Clip the L2 norm of the gradients to the given value.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )
    parser.add_argument("--snr_type", type=str, default="uniform")
    args = parser.parse_args()

    main(args)
