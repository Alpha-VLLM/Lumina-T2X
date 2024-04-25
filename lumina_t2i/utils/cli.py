import builtins
import json
import multiprocessing as mp
import os
import socket
import traceback

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist
import yaml
from torchvision.transforms.functional import to_pil_image

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


def load_model(ckpt, dtype, master_port, rank=0, num_gpus=1, is_ema=False):
    # import here to avoid huggingface Tokenizer parallelism warnings
    from diffusers.models import AutoencoderKL
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    rank0_print(f"Creating lm: {train_args.lm}")
    train_args = torch.load(os.path.join(ckpt, "model_args.pth"))
    model_lm = AutoModelForCausalLM.from_pretrained(
        train_args.lm, torch_dtype=dtype, device_map="cuda"
    )
    cap_feat_dim = model_lm.config.hidden_size
    if num_gpus > 1:
        raise NotImplementedError("Inference with >1 GPUs not yet supported")

    tokenizer = AutoTokenizer.from_pretrained(
        train_args.tokenizer_path, add_bos_token=True, add_eos_token=True
    )
    tokenizer.padding_side = "right"

    rank0_print(f"Creating vae: {train_args.vae}")
    vae = AutoencoderKL.from_pretrained(
        (
            f"stabilityai/sd-vae-ft-{train_args.vae}"
            if train_args.vae != "sdxl"
            else "stabilityai/sdxl-vae"
        ),
        torch_dtype=torch.float32,
    ).cuda()

    rank0_print(f"Creating DiT: {train_args.model}")
    # latent_size = train_args.image_size // 8
    model_dit = models.__dict__[train_args.model](
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    model_dit.eval().to("cuda", dtype=dtype)

    assert train_args.model_parallel_size == num_gpus
    ckpt = torch.load(
        os.path.join(
            ckpt,
            f"consolidated{'_ema' if is_ema else ''}.{rank:02d}-of-{num_gpus:02d}.pth",
        ),
        map_location="cpu",
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
def inference(cap, dtype, vae, config, model_dit, model_lm, tokenizer, *args, **kwargs):
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

    # sde
    sde_config = config["sde"]

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
    vae = model_config["vae"]
    model_parallel_size = model_config["model_parallel_size"]
    qk_norm = model_config["qk_norm"]
    lm = model_config["lm"]
    tokenizer_path = model_config["tokenizer_path"]

    with torch.autocast("cuda", dtype):
        while True:
            try:
                # begin sampler
                transport = create_transport(
                    path_type, prediction, loss_weight, train_eps, sample_eps
                )
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

                # tokenize user input
                cap_tok = tokenizer.encode(cap, truncation=False)
                null_cap_tok = tokenizer.encode("", truncation=False)

                tok = torch.zeros(
                    [2, max(len(cap_tok), len(null_cap_tok))],
                    dtype=torch.long,
                    device="cuda",
                )
                tok_mask = torch.zeros_like(tok, dtype=torch.bool)

                tok[0, : len(cap_tok)] = torch.tensor(cap_tok)
                tok[1, : len(null_cap_tok)] = torch.tensor(null_cap_tok)
                tok_mask[0, : len(cap_tok)] = True
                tok_mask[1, : len(null_cap_tok)] = True

                # get caption text embedding
                cap_feats = model_lm.get_decoder()(input_ids=tok).last_hidden_state

                model_kwargs = dict(
                    cap_feats=cap_feats,
                    cap_mask=tok_mask,
                    cfg_scale=cfg_scale,
                )

                if proportional_attn:
                    model_kwargs["proportional_attn"] = True
                    model_kwargs["base_seqlen"] = (image_size // 16) ** 2 + (
                        image_size // 16
                    ) * 2
                if ntk_scaling:
                    model_kwargs["ntk_factor"] = ((w // 16) * (h // 16)) / (
                        (image_size // 16) ** 2
                    )

                rank0_print(f"caption: {cap}")
                rank0_print(f"num_sampling_steps: {num_sampling_steps}")
                rank0_print(f"cfg_scale: {cfg_scale}")

                # sample noise with dit
                samples = sample_fn(z, model_dit.forward_with_cfg, **model_kwargs)[-1]
                samples = samples[:1]

                factor = 0.18215 if vae != "sdxl" else 0.13025

                # decode latent space into real image
                rank0_print(f"vae factor: {factor}")
                samples = vae.decode(samples / factor).sample
                samples = (samples + 1.0) / 2.0
                samples.clamp_(0.0, 1.0)

                return samples[0]
            except Exception:
                print(traceback.format_exc())
                return None


def main(*args, **kwargs):
    ckpt = kwargs["ckpt"]
    precision = kwargs["precision"]
    num_gpus = kwargs["num_gpus"]
    is_ema = kwargs["ema"]
    config_path = kwargs["config"]
    cap = kwargs["text"]
    output_path = kwargs["output"]

    # step 1: find available port
    master_port = find_free_port()
    # step 2: loading pretrained model with multi-gpu or not.
    rank0_print("[INFO]: loading pretrained model.")
    dtype = dtype_select(precision)
    vae, model_dit, model_lm, tokenizer, train_args = load_model(
        ckpt, dtype, master_port, 0, num_gpus, is_ema
    )
    # step 3: inference
    rank0_print("[INFO]: loading inference settings.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)[0]
        config.update({"models": train_args.__dict__})
        rank0_print(yaml.safe_dump(config))
    
    rank0_print(f"Loaded all configs: \n{json.dumps(train_args.__dict__, indent=2)}")
    
    rank0_print(f"[ATTENTION]: start inference with config: {config_path}.")
    results = inference(
        cap, dtype, config, vae, model_dit, model_lm, tokenizer, *args, **kwargs
    )

    # step 4: post processing
    rank0_print(f"[INFO]: Saving processed images.")
    img = to_pil_image(results)

    img.save(output_path)
