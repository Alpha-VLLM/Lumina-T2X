import os
import argparse
import random
import socket
import json
import math
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

import models
from transport import ODE


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
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

        prompt_embeds = text_encoder(input_ids=text_input_ids.cuda(),attention_mask=prompt_masks.cuda(),output_hidden_states=True,).hidden_states[-2]

    return prompt_embeds, prompt_masks

def none_or_str(value):
    if value == 'None':
        return None
    return value


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

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    if dist.get_rank() == 0:
        print("Loaded model arguments:",
              json.dumps(train_args.__dict__, indent=2))

    if dist.get_rank() == 0:
        print(f"Creating lm: Gemma-2B")
        
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer.padding_side = 'right'

    text_encoder = AutoModel.from_pretrained(
        "google/gemma-2b",
        torch_dtype=dtype,
        ).eval().cuda()
    # Load scheduler and models
    cap_feat_dim = text_encoder.config.hidden_size

    if dist.get_rank() == 0:
        print(f"Creating vae: {train_args.vae}")
    vae = AutoencoderKL.from_pretrained(
        (f"stabilityai/sd-vae-ft-{train_args.vae}" if train_args.vae != "sdxl" else "stabilityai/sdxl-vae"),
        torch_dtype=torch.float32,
    ).cuda()
    
    if dist.get_rank() == 0:
        print(f"Creating DiT: {train_args.model}")
    # latent_size = train_args.image_size // 8
    model = models.__dict__[train_args.model](
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
        use_flash_attn=args.use_flash_attn,
    )
    model.eval().to("cuda", dtype=dtype)

    if args.debug == False:
        # assert train_args.model_parallel_size == args.num_gpus
        if args.ema:
            print("Loading ema model.")
        ckpt = load_file(os.path.join(
            args.ckpt,
            f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.safetensors",
            )
        )
        model.load_state_dict(ckpt, strict=True)

    sample_folder_dir = args.image_save_path

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_folder_dir, 'images'), exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    info_path = os.path.join(args.image_save_path, 'data.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.loads(f.read())
        collected_id = []
        for i in info:
            collected_id.append(f'{id(i["caption"])}_{i["resolution"]}')
    else:
        info = []
        collected_id = []

    captions = []

    with open(args.caption_path, 'r', encoding='utf-8') as file:
        for line in file:
            text = line.strip()
            if text:
                captions.append(line.strip())

    ode = ODE(args.num_sampling_steps, args.solver, args.time_shifting_factor, strength=args.strength)
    
    total = len(info)
    resolution = args.resolution
    with torch.autocast("cuda", dtype):
        for res in resolution:
            for idx, caption in tqdm(enumerate(captions)):
                
                if int(args.seed) != 0:
                    torch.random.manual_seed(int(args.seed))
    
                sample_id = f'{idx}_{res.split(":")[-1]}'
                if sample_id in collected_id:
                    continue
                caps_list = [caption]
                
                res_cat, resolution = res.split(":")
                res_cat = int(res_cat)
                do_extrapolation = res_cat > 1024

                # load input image
                image = Image.open(args.image)
                res_in = resolution.split("x")
                image = image.resize((int(res_in[0]), int(res_in[1])))
                image = image.convert("RGB")
                image_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                                        inplace=True),
                ])
                image = image_transform(image).cuda()
                
                # encode input image
                with torch.no_grad():
                    vae_scale = 0.18215 if train_args.vae != "sdxl" else 0.13025
                    x1 = vae.encode(image[None]).latent_dist.sample().mul_(vae_scale)
        
                n = len(caps_list)
                w, h = resolution.split("x")
                w, h = int(w), int(h)
                latent_w, latent_h = w // 8, h // 8
                z = torch.randn([1, 4, latent_w, latent_h], device="cuda").to(dtype)
                
                # interpolate between image and noise to start timestep
                z = z * (1 - ode.t[0]) + x1 * ode.t[0]
                z = z.repeat(n * 2, 1, 1, 1)

                with torch.no_grad():
                    cap_feats, cap_mask = encode_prompt([caps_list] + [""], text_encoder, tokenizer, 0.0)

                cap_mask = cap_mask.to(cap_feats.device)

                model_kwargs = dict(
                    cap_feats=cap_feats, cap_mask=cap_mask, cfg_scale=args.cfg_scale,
                )
                    
                if args.proportional_attn:
                    model_kwargs["proportional_attn"] = True
                    model_kwargs["base_seqlen"] = (train_args.image_size // 16) ** 2
                else:
                    model_kwargs["proportional_attn"] = False
                    model_kwargs["base_seqlen"] = None

                if do_extrapolation and args.scaling_method == "Time-aware":
                    model_kwargs["scale_factor"] = math.sqrt(w * h / train_args.image_size**2)
                    model_kwargs["scale_watershed"] = args.scaling_watershed
                else:
                    model_kwargs["scale_factor"] = 1.0
                    model_kwargs["scale_watershed"] = 1.0
                
                samples = ode.sample(z, model.forward_with_cfg, **model_kwargs)[-1]
                samples = samples[:1]
                
                factor = 0.18215 if train_args.vae != "sdxl" else 0.13025
                samples = vae.decode(samples / factor).sample
                samples = (samples + 1.0) / 2.0
                samples.clamp_(0.0, 1.0)
                
                # Save samples to disk as individual .png files
                for i, (sample, cap) in enumerate(zip(samples, caps_list)):
                    img = to_pil_image(sample.float())
                    save_path = f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}.png"
                    img.save(save_path)
                    info.append({
                        'caption': cap,
                        'image_url': f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}.png",
                        'resolution': f'res: {resolution}\ntime_shift: {args.time_shifting_factor}',
                        'solver': args.solver,
                        'num_sampling_steps': args.num_sampling_steps
                    })

                with open(info_path, 'w') as f:
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
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--precision", type=str, choices=["fp32", "bf16"],
        default="bf16",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.set_defaults(ema=True)
    parser.add_argument(
        "--image_save_path", type=str, default='samples',
        help="If specified, overrides the default image save path "
             "(sample{_ema}.png in the model checkpoint directory)."
    )
    parser.add_argument(
        "--time_shifting_factor", type=float, default=1.0,
    )
    parser.add_argument(
        "--caption_path", type=str, default='prompts.txt',
    )
    parser.add_argument(
        "--resolution", type=str, default='', nargs="+",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default='',
    )
    parser.add_argument(
        "--proportional_attn", type=bool, default=True
    )
    parser.add_argument(
        "--scaling_method", type=str, default="Time-aware",
    )
    parser.add_argument(
        "--scaling_watershed", type=float, default=0.3,
    )
    parser.add_argument(
        "--use_flash_attn", type=bool, default=True
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_known_args()[0]

    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)
