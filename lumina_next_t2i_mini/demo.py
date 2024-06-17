import argparse
import builtins
import json
import math
import multiprocessing as mp
import os
import random
import socket
import traceback

import gradio as gr
import numpy as np
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image

import models
from transport import ODE


class ModelFailure:
    pass


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

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


@torch.no_grad()
def model_main(args, master_port, rank, request_queue, response_queue, mp_barrier):
    # import here to avoid huggingface Tokenizer parallelism warnings
    from diffusers.models import AutoencoderKL
    from transformers import AutoModel, AutoTokenizer

    # override the default print function since the delay can be large for child process
    original_print = builtins.print

    # Redefine the print function with flush=True by default
    def print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        original_print(*args, **kwargs)

    # Override the built-in print with the new version
    builtins.print = print

    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)

    dist.init_process_group("nccl")
    # set up fairscale environment because some methods of the Lumina model need it,
    # though for single-GPU inference fairscale actually has no effect
    torch.cuda.set_device(rank)

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    if dist.get_rank() == 0:
        print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

    if dist.get_rank() == 0:
        print(f"Creating lm: Gemma-2B")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    text_encoder = AutoModel.from_pretrained(
        "google/gemma-2b", torch_dtype=dtype, device_map="cuda", token=args.hf_token
    ).eval()
    cap_feat_dim = text_encoder.config.hidden_size
    if args.num_gpus > 1:
        raise NotImplementedError("Inference with >1 GPUs not yet supported")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=args.hf_token)
    tokenizer.padding_side = "right"

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

    if args.ema:
        print("Loading ema model.")
    ckpt = load_file(
        os.path.join(
            args.ckpt,
            f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.safetensors",
        )
    )
    model.load_state_dict(ckpt, strict=True)

    mp_barrier.wait()

    with torch.autocast("cuda", dtype):
        while True:
            (
                cap,
                neg_cap,
                resolution,
                num_sampling_steps,
                cfg_scale,
                solver,
                t_shift,
                seed,
                scaling_method,
                scaling_watershed,
                proportional_attn,
            ) = request_queue.get()

            metadata = dict(
                cap=cap,
                neg_cap=neg_cap,
                resolution=resolution,
                num_sampling_steps=num_sampling_steps,
                cfg_scale=cfg_scale,
                solver=solver,
                t_shift=t_shift,
                seed=seed,
                scaling_method=scaling_method,
                scaling_watershed=scaling_watershed,
                proportional_attn=proportional_attn,
            )
            print("> params:", json.dumps(metadata, indent=2))

            try:
                do_extrapolation = "Extrapolation" in resolution
                resolution = resolution.split(" ")[-1]
                w, h = resolution.split("x")
                w, h = int(w), int(h)
                latent_w, latent_h = w // 8, h // 8
                if int(seed) != 0:
                    torch.random.manual_seed(int(seed))
                z = torch.randn([1, 4, latent_h, latent_w], device="cuda").to(dtype)
                z = z.repeat(2, 1, 1, 1)

                with torch.no_grad():
                    if neg_cap != "":
                        cap_feats, cap_mask = encode_prompt([cap] + [neg_cap], text_encoder, tokenizer, 0.0)
                    else:
                        cap_feats, cap_mask = encode_prompt([cap] + [""], text_encoder, tokenizer, 0.0)

                cap_mask = cap_mask.to(cap_feats.device)

                model_kwargs = dict(
                    cap_feats=cap_feats,
                    cap_mask=cap_mask,
                    cfg_scale=cfg_scale,
                )
                if proportional_attn:
                    model_kwargs["proportional_attn"] = True
                    model_kwargs["base_seqlen"] = (train_args.image_size // 16) ** 2
                else:
                    model_kwargs["proportional_attn"] = False
                    model_kwargs["base_seqlen"] = None

                if do_extrapolation and scaling_method == "Time-aware":
                    model_kwargs["scale_factor"] = math.sqrt(w * h / train_args.image_size**2)
                    model_kwargs["scale_watershed"] = scaling_watershed
                else:
                    model_kwargs["scale_factor"] = 1.0
                    model_kwargs["scale_watershed"] = 1.0

                if dist.get_rank() == 0:
                    print(f"> caption: {cap}")
                    print(f"> num_sampling_steps: {num_sampling_steps}")
                    print(f"> cfg_scale: {cfg_scale}")

                print("> start sample")
                samples = ODE(num_sampling_steps, solver, t_shift).sample(z, model.forward_with_cfg, **model_kwargs)[-1]
                samples = samples[:1]

                factor = 0.18215 if train_args.vae != "sdxl" else 0.13025
                print(f"> vae factor: {factor}")
                samples = vae.decode(samples / factor).sample
                samples = (samples + 1.0) / 2.0
                samples.clamp_(0.0, 1.0)

                img = to_pil_image(samples[0].float())
                print("> generated image, done.")

                if response_queue is not None:
                    response_queue.put((img, metadata))

            except Exception:
                print(traceback.format_exc())
                response_queue.put(ModelFailure())


def none_or_str(value):
    if value == "None":
        return None
    return value


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--hf_token", type=str, default=None, help="huggingface read token for accessing gated repo.")
    parser.add_argument("--use_flash_attn", type=bool, default=True, help="Use flash attention for the model.")

    args = parser.parse_known_args()[0]

    if args.num_gpus != 1:
        raise NotImplementedError("Multi-GPU Inference is not yet supported")

    master_port = find_free_port()

    processes = []
    request_queues = []
    response_queue = mp.Queue()
    mp_barrier = mp.Barrier(args.num_gpus + 1)
    for i in range(args.num_gpus):
        request_queues.append(mp.Queue())
        p = mp.Process(
            target=model_main,
            args=(
                args,
                master_port,
                i,
                request_queues[i],
                response_queue if i == 0 else None,
                mp_barrier,
            ),
        )
        p.start()
        processes.append(p)

    description = """
    # Lumina Next Text-to-Image

    Lumina-Next-T2I is a 2B Next-DiT model with 2B text encoder.

    Demo current model: `Lumina-Next-T2I`

    """
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                cap = gr.Textbox(
                    lines=2,
                    label="Caption",
                    interactive=True,
                    value="Miss Mexico portrait of the most beautiful mexican woman, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8, "
                    "ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, facial-symmetry, 8k",
                    placeholder="Enter a caption.",
                )
                neg_cap = gr.Textbox(
                    lines=2,
                    label="Negative Caption",
                    interactive=True,
                    value="",
                    placeholder="Enter a negative caption.",
                )
                with gr.Row():
                    res_choices = [
                        "1024x1024",
                        "512x2048",
                        "2048x512",
                        "(Extrapolation) 1536x1536",
                        "(Extrapolation) 2048x1024",
                        "(Extrapolation) 1024x2048",
                        "(Extrapolation) 2048x2048",
                        "(Extrapolation) 4096x1024",
                        "(Extrapolation) 1024x4096",
                    ]
                    resolution = gr.Dropdown(value=res_choices[0], choices=res_choices, label="Resolution")
                with gr.Row():
                    num_sampling_steps = gr.Slider(
                        minimum=1,
                        maximum=70,
                        value=30,
                        step=1,
                        interactive=True,
                        label="Sampling steps",
                    )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=int(1e5),
                        value=1,
                        step=1,
                        interactive=True,
                        label="Seed (0 for random)",
                    )
                with gr.Row():
                    solver = gr.Dropdown(
                        value="midpoint",
                        choices=["euler", "midpoint", "rk4"],
                        label="solver",
                    )
                    t_shift = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=4,
                        step=1,
                        interactive=True,
                        label="Time shift",
                    )
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=4.0,
                        interactive=True,
                        label="CFG scale",
                    )
                with gr.Accordion("Advanced Settings for Resolution Extrapolation", open=False):
                    with gr.Row():
                        scaling_method = gr.Dropdown(
                            value="Time-aware",
                            choices=["Time-aware", "None"],
                            label="RoPE scaling method",
                        )
                        scaling_watershed = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            interactive=True,
                            label="Linear/NTK watershed",
                        )
                    with gr.Row():
                        proportional_attn = gr.Checkbox(
                            value=True,
                            interactive=True,
                            label="Proportional attention",
                        )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column():
                output_img = gr.Image(
                    label="Generated image",
                    interactive=False,
                    format="png",
                )
                with gr.Accordion(label="Generation Parameters", open=True):
                    gr_metadata = gr.JSON(label="metadata", show_label=False)

        with gr.Row():
            gr.Examples(
                [
                    ["👽🤖👹👻"],
                    ["孤舟蓑笠翁"],
                    ["两只黄鹂鸣翠柳"],
                    ["大漠孤烟直，长河落日圆"],
                    ["秋风起兮白云飞，草木黄落兮雁南归"],
                    ["도쿄 타워, 최고 품질의 우키요에, 에도 시대"],
                    ["味噌ラーメン, 最高品質の浮世絵、江戸時代。"],
                    ["東京タワー、最高品質の浮世絵、江戸時代。"],
                    ["Astronaut on Mars During sunset"],
                    ["Tour de Tokyo, estampes ukiyo-e de la plus haute qualité, période Edo"],
                    ["🐔 playing 🏀"],
                    ["☃️ with 🌹 in the ❄️"],
                    ["🐶 wearing 😎  flying on 🌈 "],
                    ["A small 🍎 and 🍊 with 😁 emoji in the Sahara desert"],
                    ["Токийская башня, лучшие укиё-э, период Эдо"],
                    ["Tokio-Turm, hochwertigste Ukiyo-e, Edo-Zeit"],
                    ["A scared cute rabbit in Happy Tree Friends style and punk vibe."],  # noqa
                    ["A humanoid eagle soldier of the First World War."],  # noqa
                    [
                        "A cute Christmas mockup on an old wooden industrial desk table with Christmas decorations and bokeh lights in the background."
                    ],
                    [
                        "A front view of a romantic flower shop in France filled with various blooming flowers including lavenders and roses."
                    ],
                    ["An old man, portrayed as a retro superhero, stands in the streets of New York City at night"],
                    [
                        "many trees are surrounded by a lake in autumn colors, in the style of nature-inspired imagery, havencore, brightly colored, dark white and dark orange, bright primary colors, environmental activism, forestpunk --ar 64:51"
                    ],
                    [
                        "A fluffy mouse holding a watermelon, in a magical and colorful setting, illustrated in the style of Hayao Miyazaki anime by Studio Ghibli."
                    ],
                    [
                        "Inka warrior with a war make up, medium shot, natural light, Award winning wildlife photography, hyperrealistic, 8k resolution, --ar 9:16"
                    ],
                    [
                        "Character of lion in style of saiyan, mafia, gangsta, citylights background, Hyper detailed, hyper realistic, unreal engine ue5, cgi 3d, cinematic shot, 8k"
                    ],
                    [
                        "In the sky above, a giant, whimsical cloud shaped like the 😊 emoji casts a soft, golden light over the scene"
                    ],
                    [
                        "Cyberpunk eagle, neon ambiance, abstract black oil, gear mecha, detailed acrylic, grunge, intricate complexity, rendered in unreal engine 5, photorealistic, 8k"
                    ],
                    [
                        "close-up photo of a beautiful red rose breaking through a cube made of ice , splintered cracked ice surface, frosted colors, blood dripping from rose, melting ice, Valentine’s Day vibes, cinematic, sharp focus, intricate, cinematic, dramatic light"
                    ],
                    [
                        "3D cartoon Fox Head with Human Body, Wearing Iridescent Holographic Liquid Texture & Translucent Material Sun Protective Shirt, Boss Feel, Nike or Addidas Sun Protective Shirt, WitchPunk, Y2K Style, Green and blue, Blue, Metallic Feel, Strong Reflection, plain background, no background, pure single color background, Digital Fashion, Surreal Futurism, Supreme Kong NFT Artwork Style, disney style, headshot photography for portrait studio shoot, fashion editorial aesthetic, high resolution in the style of HAPE PRIME NFT, NFT 3D IP Feel, Bored Ape Yacht Club NFT project Feel, high detail, fine luster, 3D render, oc render, best quality, 8K, bright, front lighting, Face Shot, fine luster, ultra detailed"
                    ],
                ],
                [cap],
                label="Examples",
            )

        def on_submit(*args):
            for q in request_queues:
                q.put(args)
            result = response_queue.get()
            if isinstance(result, ModelFailure):
                raise RuntimeError
            img, metadata = result

            return img, metadata

        submit_btn.click(
            on_submit,
            [
                cap,
                neg_cap,
                resolution,
                num_sampling_steps,
                cfg_scale,
                solver,
                t_shift,
                seed,
                scaling_method,
                scaling_watershed,
                proportional_attn,
            ],
            [output_img, gr_metadata],
        )

        def show_scaling_watershed(scaling_m):
            return gr.update(visible=scaling_m == "Time-aware")

        scaling_method.change(show_scaling_watershed, scaling_method, scaling_watershed)

    mp_barrier.wait()
    demo.queue().launch(
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
