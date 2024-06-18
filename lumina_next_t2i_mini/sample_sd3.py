import argparse
import json
import os
import random
import socket
import time

from diffusers import StableDiffusion3Pipeline
import numpy as np
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import models
from transport import ODE


def main(args, rank, master_port):
    # Setup PyTorch:
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=dtype
    ).to("cuda")

    if dist.get_rank() == 0:
        print("Loaded SD3 pipeline")

    vae = pipe.vae
    model = pipe.transformer
    model.eval().to("cuda", dtype=dtype)

    sample_folder_dir = args.image_save_path

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_folder_dir, "images"), exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    info_path = os.path.join(args.image_save_path, "data.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.loads(f.read())
        collected_id = []
        for i in info:
            collected_id.append(f'{id(i["caption"])}_{i["resolution"]}')
    else:
        info = []
        collected_id = []

    captions = []

    with open(args.caption_path, "r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                captions.append(line.strip())

    total = len(info)
    resolution = args.resolution
    for res in resolution:
        for idx, caption in tqdm(enumerate(captions)):

            if int(args.seed) != 0:
                torch.random.manual_seed(int(args.seed))

            sample_id = f'{idx}_{res.split(":")[-1]}'
            if sample_id in collected_id:
                continue

            res_cat, resolution = res.split(":")
            res_cat = int(res_cat)
            caps_list = [caption]

            n = 1
            w, h = resolution.split("x")
            w, h = int(w), int(h)
            latent_w, latent_h = w // 8, h // 8
            z = torch.randn([n, 16, latent_w, latent_h], device="cuda").to(dtype)
            z = z.repeat(n * 2, 1, 1, 1)

            with torch.no_grad():
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                    pipe.encode_prompt(prompt=caption, prompt_2=caption, prompt_3=caption, device="cuda")
                )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

            model_kwargs = dict(
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                cfg_scale=args.cfg_scale,
                return_dict=False,
            )

            samples = ODE(args.num_sampling_steps, args.solver, args.time_shifting_factor, use_sd3=True).sample(
                z, model.forward, **model_kwargs
            )[-1]
            samples = samples[:1]

            samples = vae.decode(samples / 1.5305 + 0.0609).sample
            samples = (samples + 1.0) / 2.0
            samples.clamp_(0.0, 1.0)

            # Save samples to disk as individual .png files
            for i, (sample, cap) in enumerate(zip(samples, caps_list)):
                img = to_pil_image(sample.float())
                save_path = f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}.png"
                img.save(save_path)
                info.append(
                    {
                        "caption": cap,
                        "image_url": f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}.png",
                        "resolution": f"res: {resolution}\ntime_shift: {args.time_shifting_factor}",
                        "solver": args.solver,
                        "num_sampling_steps": args.num_sampling_steps,
                    }
                )

            with open(info_path, "w") as f:
                f.write(json.dumps(info))

            total += len(samples)
            dist.barrier()

    dist.barrier()
    dist.barrier()
    dist.destroy_process_group()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default="bf16",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.set_defaults(ema=True)
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="samples",
        help="If specified, overrides the default image save path "
        "(sample{_ema}.png in the model checkpoint directory).",
    )
    parser.add_argument(
        "--time_shifting_factor",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="prompts.txt",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="",
        nargs="+",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
    )
    parser.add_argument("--proportional_attn", type=bool, default=True)
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="Time-aware",
    )
    parser.add_argument(
        "--scaling_watershed",
        type=float,
        default=0.3,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_known_args()[0]

    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)
