import argparse
import builtins
import json
import math
import multiprocessing as mp
import os
import random
import socket
import traceback

import fairscale.nn.model_parallel.initialize as fs_init
import gradio as gr
import numpy as np
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image

import models
from transport import Sampler, create_transport


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
    fs_init.initialize_model_parallel(args.num_gpus)
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
    )
    model.eval().to("cuda", dtype=dtype)

    assert train_args.model_parallel_size == args.num_gpus
    if args.ema:
        print(args.ckpt, "Loading ema model.")
    ckpt = load_file(
        os.path.join(
            args.ckpt,
            f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.safetensors",
        ),
        device="cpu",
    )
    model.load_state_dict(ckpt, strict=True)

    mp_barrier.wait()

    with torch.autocast("cuda", dtype):
        while True:
            (
                cap1,
                cap2,
                cap3,
                cap4,
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
                cap1=cap1,
                cap2=cap2,
                cap3=cap3,
                cap4=cap4,
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
                # begin sampler
                transport = create_transport(
                    args.path_type,
                    args.prediction,
                    args.loss_weight,
                    args.train_eps,
                    args.sample_eps,
                )
                sampler = Sampler(transport)
                sample_fn = sampler.sample_ode(
                    sampling_method=solver,
                    num_steps=num_sampling_steps,
                    atol=args.atol,
                    rtol=args.rtol,
                    reverse=args.reverse,
                    time_shifting_factor=t_shift,
                )
                # end sampler

                do_extrapolation = "Extrapolation" in resolution
                split = resolution.split(" ")[1].replace("(", "")
                w_split, h_split = split.split("x")
                resolution = resolution.split(" ")[0]
                w, h = resolution.split("x")
                w, h = int(w), int(h)
                latent_w, latent_h = w // 8, h // 8
                if int(seed) != 0:
                    torch.random.manual_seed(int(seed))
                z = torch.randn([1, 4, latent_h, latent_w], device="cuda").to(dtype)
                z = z.repeat(2, 1, 1, 1)

                cap_list = [cap1, cap2, cap3, cap4]
                global_cap = " ".join(cap_list)
                with torch.no_grad():
                    if neg_cap != "":
                        cap_feats, cap_mask = encode_prompt(
                            cap_list + [neg_cap] + [global_cap], text_encoder, tokenizer, 0.0
                        )
                    else:
                        cap_feats, cap_mask = encode_prompt(
                            cap_list + [""] + [global_cap], text_encoder, tokenizer, 0.0
                        )

                cap_mask = cap_mask.to(cap_feats.device)

                model_kwargs = dict(
                    cap_feats=cap_feats[:-1],
                    cap_mask=cap_mask[:-1],
                    global_cap_feats=cap_feats[-1:],
                    global_cap_mask=cap_mask[-1:],
                    cfg_scale=cfg_scale,
                    h_split_num=int(h_split),
                    w_split_num=int(w_split),
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
                    print(f"> caption: {global_cap}")
                    print(f"> num_sampling_steps: {num_sampling_steps}")
                    print(f"> cfg_scale: {cfg_scale}")

                print("> start sample")
                samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
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


def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument(
        "--path-type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "VP"],
        help="the type of path for transport: 'Linear', 'GVP' (Geodesic Vector Pursuit), or 'VP' (Vector Pursuit).",
    )
    group.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
        help="the prediction model for the transport dynamics.",
    )
    group.add_argument(
        "--loss-weight",
        type=none_or_str,
        default=None,
        choices=[None, "velocity", "likelihood"],
        help="the weighting of different components in the loss function, can be 'velocity' for dynamic modeling, 'likelihood' for statistical consistency, or None for no weighting.",
    )
    group.add_argument("--sample-eps", type=float, help="sampling in the transport model.")
    group.add_argument("--train-eps", type=float, help="training to stabilize the learning process.")


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the ODE solver.",
    )
    group.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the ODE solver.",
    )
    group.add_argument("--reverse", action="store_true", help="run the ODE solver in reverse.")
    group.add_argument(
        "--likelihood",
        action="store_true",
        help="Enable calculation of likelihood during the ODE solving process.",
    )


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

    parse_transport_args(parser)
    parse_ode_args(parser)

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
    # Lumina Next Text-to-Image (Compositional Generation)
    <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 7; padding-right: 20px;">
            <h3>Lumina-Next-T2I is a 2B Next-DiT model with 2B text encoder.</h3>
            <p>Demo current model: `Lumina-Next-T2I`</p>
            <h3>Lumina-Next-T2I Compositional Generation creates images in a grid format, specifying a caption for each grid to represent the style of different regions.</h3>
            <h3><span style='color: orange;'>For example, a 4x1 grid size means the width is divided into 4 grids and the height is divided into 1 grid.</h3>
        </div>
        <div style="flex: 5; padding-left: 20px;">
            <img src="/file=../assets/compositional_intro.png" width="90%"/>
        </div>
    </div>
    """
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(description, latex_delimiters=[])
        with gr.Row():
            with gr.Column():
                cap1 = gr.Textbox(
                    lines=2,
                    label="Caption (Grid #1)",
                    interactive=True,
                    value="A full moon hanging in the sky.",
                    placeholder="Enter a caption.",
                )
                cap2 = gr.Textbox(
                    lines=2,
                    label="Caption (Grid #2)",
                    interactive=True,
                    value="A small river meanders by, sparkling with light.",
                    placeholder="Enter a caption.",
                )
                cap3 = gr.Textbox(
                    lines=2,
                    label="Caption (Grid #3)",
                    interactive=True,
                    value="An old house with old brick walls.",
                    placeholder="Enter a caption.",
                )
                cap4 = gr.Textbox(
                    lines=2,
                    label="Caption (Grid #4)",
                    interactive=True,
                    value="A huge cherry tree with pink cherry blossoms.",
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
                        "2048x1024 (4x1 Grids)",
                        "2560x1024 (4x1 Grids)",
                        "3072x1024 (4x1 Grids)",
                        "1024x1024 (2x2 Grids)",
                        "1536x1536 (2x2 Grids)",
                        "2048x2048 (2x2 Grids)",
                        "1024x2048 (1x4 Grids)",
                        "1024x2560 (1x4 Grids)",
                        "1024x3072 (1x4 Grids)",
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
                    [
                        "A colossal ancient robot stands amidst the ruins of a forgotten civilization. Its metallic body is covered in intricate carvings and symbols, showing signs of age and wear.",
                        "A colossal ancient robot stands amidst the ruins of a forgotten civilization. Its metallic body is covered in intricate carvings and symbols, showing signs of age and wear. ",
                        "A winding countryside path meanders through rolling green hills, lined with wildflowers and tall grasses swaying in the breeze.",
                        "A quaint countryside cottage bathed in the warm glow of the setting sun. The small house is surrounded by a lush garden filled with blooming flowers and tall, swaying grass.",
                        "2048x1024 (4x1 Grids)",
                    ],
                    [
                        "A full moon hanging in the sky.",
                        "A few small boats appeared on the lake.",
                        "Hogwarts' Castle in the Moonlight.",
                        "The calm surface of the lake is illuminated by the moonlight.",
                        "3072x1024 (4x1 Grids)",
                    ],
                    [
                        "A tranquil cherry blossom forest with pink petals covering the ground.",
                        "A majestic polar bear stands on a vast, snowy landscape, its white fur blending seamlessly with the snow.",
                        "A majestic polar bear stands on a vast, snowy landscape, its white fur blending seamlessly with the snow.",
                        "A tranquil cherry blossom forest with pink petals covering the ground.",
                        "2048x1024 (4x1 Grids)",
                    ],
                    [
                        "A close-up portrait of a grey Maine cat with striking, emerald blue eyes that reflect curiosity and wisdom. The cat has soft, fluffy fur with a mix of white and gray tabby markings.",
                        "A close-up portrait of a grey Maine cat with striking, emerald blue eyes that reflect curiosity and wisdom. The cat has soft, fluffy fur with a mix of white and gray tabby markings.",
                        "A sharply dressed man in a tailored, dark navy suit stands confidently. The suit jacket fits perfectly, with a crisp white dress shirt underneath and a silk tie in a subtle, elegant pattern",
                        "A sharply dressed man in a tailored, dark navy suit stands confidently. The suit jacket fits perfectly, with a crisp white dress shirt underneath and a silk tie in a subtle, elegant pattern",
                        "1536x1536 (2x2 Grids)",
                    ],
                    [
                        "An underwater city inhabited by aquatic creatures, with colorful coral reefs and schools of fish.",
                        "A sprawling space station, bustling with activity and interstellar travelers.",
                        "A dystopian wasteland with ruins and debris.",
                        "A lone astronaut exploring the desolate surface of a distant planet, with the vast expanse of space stretching out behind them.",
                        "1536x1536 (2x2 Grids)",
                    ],
                    [
                        "A snowy mountain.",
                        "A steampunk ship sails gracefully",
                        "A beautiful river",
                        "A tranquil garden filled with blooming flowers",
                        "1024x2048 (1x4 Grids)",
                    ],
                ],
                [cap1, cap2, cap3, cap4, resolution],
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
                cap1,
                cap2,
                cap3,
                cap4,
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
        allowed_paths=["../assets"],
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
