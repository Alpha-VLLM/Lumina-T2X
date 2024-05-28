#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import argparse
import json
import os
import socket
import sys

from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist
from torchvision.utils import save_image

import models
from transport import Sampler, create_transport


def none_or_str(value):
    if value == "None":
        return None
    return value


def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--sampling-method",
        type=str,
        default="dopri5",
        help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq",
    )
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")


def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument(
        "--diffusion-form",
        type=str,
        default="sigma",
        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],
        help="form of diffusion coefficient in the SDE",
    )
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument(
        "--last-step",
        type=none_or_str,
        default="Mean",
        choices=[None, "Mean", "Tweedie", "Euler"],
        help="form of last step taken in the SDE",
    )
    group.add_argument("--last-step-size", type=float, default=0.04, help="size of the last step taken")


def main(args, rank, master_port):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    dist.init_process_group("nccl")
    fs_init.initialize_model_parallel(args.num_gpus)
    torch.cuda.set_device(rank)

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))

    if dist.get_rank() == 0:
        print("Model arguments used for inference:", json.dumps(train_args.__dict__, indent=2))

    # Load model:
    image_size = train_args.image_size
    # latent_size = image_size // 8
    latent_size = image_size // 4
    model = models.__dict__[train_args.model](
        input_size=latent_size,
        num_classes=train_args.num_classes,
        qk_norm=train_args.qk_norm,
    )

    torch_dtype = {
        "fp32": torch.float,
        "tf32": torch.float,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.precision]
    model.to(torch_dtype).cuda()
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # assert train_args.model_parallel_size == args.num_gpus
    ckpt = torch.load(
        os.path.join(
            args.ckpt,
            f"consolidated{'_ema' if args.ema else ''}." f"{rank:02d}-of-{args.num_gpus:02d}.pth",
        ),
        map_location="cpu",
    )
    model.load_state_dict(ckpt, strict=True)

    model.eval()  # important!

    transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    sampler = Sampler(transport)

    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse,
            )

    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{train_args.vae}"
        if args.local_diffusers_model_root is None
        else os.path.join(args.local_diffusers_model_root, f"stabilityai/sd-vae-ft-{train_args.vae}")
    ).cuda()

    # Create sampling noise:
    n = len(args.class_labels)
    z = torch.randn(
        n,
        4,
        latent_size,
        latent_size,
        dtype=torch_dtype,
        device="cuda",
    )
    y = torch.tensor(args.class_labels, device="cuda")

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device="cuda")
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    if rank == 0:
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        # Save and display images:
        save_image(
            samples,
            args.image_save_path or os.path.join(args.ckpt, f"sample{'_ema' if args.ema else ''}.png"),
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
        )

    dist.barrier()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = sys.argv[1]
    if mode not in ["ODE", "SDE"]:
        mode = "ODE"

    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--class_labels",
        type=int,
        nargs="+",
        help="Class labels to generate the images for.",
        default=[207, 360, 387, 974, 88, 979, 417, 279],
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "tf32", "fp16", "bf16"],
        default="tf32",
    )
    parser.add_argument(
        "--local_diffusers_model_root",
        type=str,
        help="Specify the root directory if diffusers models are to be loaded "
        "from the local filesystem (instead of being automatically "
        "downloaded from the Internet). Useful in environments without "
        "Internet access.",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.add_argument("--no_ema", action="store_false", dest="ema", help="Do not use EMA models.")
    parser.set_defaults(ema=True)
    parser.add_argument(
        "--image_save_path",
        type=str,
        help="If specified, overrides the default image save path "
        "(sample{_ema}.png in the model checkpoint directory).",
    )

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    # args = parser.parse_args()
    args = parser.parse_known_args()[0]

    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."
    main(args, 0, master_port)
