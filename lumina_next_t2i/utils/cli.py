import builtins
import math
import os
import random
import socket
import time
import traceback
import warnings

import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
import yaml

from .. import models
from ..transport import Sampler, create_transport


def rank0_print(*text):
    if dist.get_rank() == 0:
        print(*text)


def flush_print():
    # override the default print function since the delay can be large for child process
    original_print = builtins.print

    # Redefine the print function with flush=True by default
    def print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        original_print(*args, **kwargs)

    return print


def dtype_select(precision):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    return dtype[precision]


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

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


def load_model(
    ckpt,
    dtype,
    master_port,
    rank=0,
    num_gpus=1,
    is_ema=False,
    token: str | bool = False,
    ckpt_lm=None,
):
    # import here to avoid huggingface Tokenizer parallelism warnings
    from diffusers.models import AutoencoderKL
    from transformers import AutoModel, AutoTokenizer

    if num_gpus != 1:
        raise NotImplementedError("Multi-GPU Inference is not yet supported")

    # setup multi-processing
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(num_gpus)

    dist.init_process_group("nccl")

    fs_init.initialize_model_parallel(num_gpus)
    torch.cuda.set_device(rank)

    train_args = torch.load(os.path.join(ckpt, "model_args.pth"))

    if ckpt_lm is None:
        rank0_print(f"> Creating LLM from train_args: {train_args.lm}")
        if not token:
            warnings.warn(
                "> Attention! You need to input your access token in the huggingface when loading the gated repo, "
                "or use the `huggingface-cli login` (stored in ~/.huggingface by default) to log in."
            )
        ckpt_lm = train_args.lm

    rank0_print(f"> Creating LLM model.")
    model_lm = AutoModel.from_pretrained(ckpt_lm, torch_dtype=dtype, device_map="cuda", token=token)
    cap_feat_dim = model_lm.config.hidden_size
    if num_gpus > 1:
        raise NotImplementedError("Inference with >1 GPUs not yet supported")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_lm, token=token)
    tokenizer.padding_side = "right"

    rank0_print(f"> Creating VAE model: {train_args.vae}")
    vae = AutoencoderKL.from_pretrained(
        (f"stabilityai/sd-vae-ft-{train_args.vae}" if train_args.vae != "sdxl" else "stabilityai/sdxl-vae"),
        torch_dtype=torch.float32,
    ).cuda()

    rank0_print(f"> Creating DiT model: {train_args.model}")
    # latent_size = train_args.image_size // 8
    model_dit = models.__dict__[train_args.model](
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    model_dit.eval().to("cuda", dtype=dtype)

    assert train_args.model_parallel_size == num_gpus
    if is_ema:
        print("Loading ema model.")
    ckpt = load_file(
        os.path.join(
            ckpt,
            f"consolidated{'_ema' if is_ema else ''}.{rank:02d}-of-{num_gpus:02d}.safetensors",
        ),
    )
    model_dit.load_state_dict(ckpt, strict=True)

    return vae, model_dit, model_lm, tokenizer, train_args


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@torch.no_grad()
def inference(cap, dtype, config, vae, model_dit, text_encoder, tokenizer, *args, **kwargs):
    # transport
    transport_config = config["transport"]
    path_type = transport_config["path_type"]
    prediction = transport_config["prediction"]
    loss_weight = transport_config["loss_weight"]
    train_eps = transport_config["train_eps"]
    sample_eps = transport_config["sample_eps"]
    # ode
    ode_config = config["ode"]
    atol = ode_config["atol"]
    rtol = ode_config["rtol"]
    reverse = ode_config["reverse"]
    likelihood = ode_config["likelihood"]

    # inference
    infer_config = config["infer"]
    resolution = infer_config["resolution"]
    num_sampling_steps = infer_config["num_sampling_steps"]
    cfg_scale = infer_config["cfg_scale"]
    solver = infer_config["solver"]
    t_shift = infer_config["t_shift"]
    seed = infer_config["seed"]
    ntk_scaling = infer_config["ntk_scaling"]
    proportional_attn = infer_config["proportional_attn"]

    # model config
    model_config = config["models"]
    model_name = model_config["model"]
    image_size = model_config["image_size"]
    vae_type = model_config["vae"]

    with torch.autocast("cuda", dtype):
        while True:
            try:
                # begin sampler
                transport = create_transport(path_type, prediction, loss_weight, train_eps, sample_eps)
                sampler = Sampler(transport)
                sample_fn = sampler.sample_ode(
                    sampling_method=solver,
                    num_steps=num_sampling_steps,
                    atol=atol,
                    rtol=rtol,
                    reverse=reverse,
                    time_shifting_factor=t_shift,
                )
                # end sampler

                # getting resolution setting
                resolution = resolution.split(" ")[-1]
                w, h = resolution.split("x")
                w, h = int(w), int(h)
                latent_w, latent_h = w // 8, h // 8

                if int(seed) != 0:
                    torch.random.manual_seed(int(seed))

                # initialize latent space
                z = torch.randn([1, 4, latent_h, latent_w], device="cuda").to(dtype)
                z = z.repeat(2, 1, 1, 1)

                with torch.no_grad():
                    cap_feats, cap_mask = encode_prompt([cap] + [""], text_encoder, tokenizer, 0.0)
                # get caption text embedding
                cap_mask = cap_mask.to(cap_feats.device)

                model_kwargs = dict(
                    cap_feats=cap_feats,
                    cap_mask=cap_mask,
                    cfg_scale=cfg_scale,
                )
                if proportional_attn:
                    model_kwargs["proportional_attn"] = True
                    model_kwargs["base_seqlen"] = (image_size // 16) ** 2
                if ntk_scaling:
                    model_kwargs["ntk_factor"] = math.sqrt(w * h / image_size**2)

                rank0_print(f"> Caption: {cap}")
                rank0_print(f"> Num_sampling_steps: {num_sampling_steps}")
                rank0_print(f"> Cfg_scale: {cfg_scale}")

                # sample noise with dit
                samples = sample_fn(z, model_dit.forward_with_cfg, **model_kwargs)[-1]
                samples = samples[:1]

                factor = 0.18215 if vae_type != "sdxl" else 0.13025

                # decode latent space into real image
                rank0_print(f"> VAE factor: {factor}")
                samples = vae.decode(samples / factor).sample
                samples = (samples + 1.0) / 2.0
                samples.clamp_(0.0, 1.0)

                return samples[0]
            except Exception:
                print(traceback.format_exc())
                return None


def main(
    num_gpus,
    ckpt,
    ckpt_lm,
    is_ema,
    precision,
    config_path,
    token,
    cap,
    output_path,
    *args,
    **kwargs,
):
    # step 1: find available port
    master_port = find_free_port()
    # step 2: loading pretrained model with multi-gpu or not.
    print("> loading inference settings.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)[0]
    # if user do not pass any parameter about model ckpt, using yaml config.
    model_config = config["model"]
    # parameter from cli
    if ckpt is None or ckpt_lm is None or token is None:
        if (
            model_config.get("ckpt", None) is None
            or model_config.get("ckpt_lm", None) is None
            or model_config.get("token", None) is None
        ):
            raise ValueError(
                "please setting correct model path in yaml config, or pass `--ckpt`"
                "`--ckpt`, `--token` as cli options."
            )
        ckpt = model_config["ckpt"]
        ckpt_lm = model_config["ckpt_lm"]
        token = model_config["token"]
    else:
        print("> loading model path from cli options.")

    print("> loading pretrained model.")
    dtype = dtype_select(precision)
    # init distributed state
    vae, model_dit, model_lm, tokenizer, train_args = load_model(
        ckpt, dtype, master_port, 0, num_gpus, is_ema, token, ckpt_lm
    )
    config.update({"models": train_args.__dict__})
    rank0_print(yaml.safe_dump(config))

    # step 3: inference
    rank0_print(f"> [ATTENTION] start inference with config: {config_path}.")
    results = inference(cap, dtype, config, vae, model_dit, model_lm, tokenizer, *args, **kwargs)

    # step 4: post processing
    rank0_print(f"> Saving processed images.")
    img = to_pil_image(results.float())

    os.makedirs(output_path, exist_ok=True)
    if os.path.isdir(output_path):
        rank0_print(f"> Image saved in {output_path}.")
    img_name = "_".join(cap.split(" ")).split(".")[0]
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    img.save(os.path.join(output_path, f"{img_name}_{current_time}_lumina.png"))
