import argparse
import builtins
import json
import multiprocessing as mp
import os
import socket
import sys
import traceback

import fairscale.nn.model_parallel.initialize as fs_init
import gradio as gr
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image

import models
from transport import create_transport, Sampler


class ModelFailure: pass


@torch.no_grad()
def model_main(args, master_port, rank, request_queue, response_queue, mp_barrier):

    # import here to avoid huggingface Tokenizer parallelism warnings
    from diffusers.models import AutoencoderKL
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # override the default print function since the delay can be large for child process
    original_print = builtins.print
    # Redefine the print function with flush=True by default
    def print(*args, **kwargs):
        kwargs.setdefault('flush', True)
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
        print(
            "Loaded model arguments:",
            json.dumps(train_args.__dict__, indent=2)
        )

    if dist.get_rank() == 0:
        print(f"Creating lm: {train_args.lm}")
    model_lm = AutoModelForCausalLM.from_pretrained(train_args.lm).cuda().eval().bfloat16()  # meta-llama/Llama-2-7b-hf
    cap_feat_dim = model_lm.config.hidden_size
    if args.num_gpus > 1:
        raise NotImplementedError("Inference with >1 GPUs not yet supported")

    tokenizer = AutoTokenizer.from_pretrained(train_args.tokenizer_path, add_bos_token=True, add_eos_token=True)
    tokenizer.padding_side = 'right'

    if dist.get_rank() == 0:
        print(f"Creating vae: {train_args.vae}")
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{train_args.vae}"
        if train_args.vae != "sdxl"
        else "stabilityai/sdxl-vae"
    ).cuda().eval().bfloat16()

    if dist.get_rank() == 0:
        print(f"Creating DiT: {train_args.model}")
    # latent_size = train_args.image_size // 8
    model = models.__dict__[train_args.model](
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    model.cuda().eval().bfloat16()

    assert train_args.model_parallel_size == args.num_gpus
    ckpt = torch.load(os.path.join(
        args.ckpt, f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.pth"
    ), map_location="cpu")
    model.load_state_dict(ckpt, strict=True)

    mp_barrier.wait()

    while True:
        caption, resolution, num_sampling_steps, cfg_scale, solver_method = request_queue.get()

        try:
            # begin sampler
            transport = create_transport(
                args.path_type,
                args.prediction,
                args.loss_weight,
                args.train_eps,
                args.sample_eps
            )
            sampler = Sampler(transport)
            if args.sampler_mode == "ODE":
                if args.likelihood:
                    # assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
                    sample_fn = sampler.sample_ode_likelihood(
                        sampling_method=solver_method,
                        num_steps=num_sampling_steps,
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                else:
                    sample_fn = sampler.sample_ode(
                        sampling_method=solver_method,
                        num_steps=num_sampling_steps,
                        atol=args.atol,
                        rtol=args.rtol,
                        reverse=args.reverse
                    )
            elif args.sampler_mode == "SDE":
                sample_fn = sampler.sample_sde(
                    sampling_method=solver_method,
                    diffusion_form=args.diffusion_form,
                    diffusion_norm=args.diffusion_norm,
                    last_step=args.last_step,
                    last_step_size=args.last_step_size,
                    num_steps=num_sampling_steps,
                )
            else:
                raise ValueError(f"Unknown sampler {args.sampler_mode}")
            # end sampler

            w, h = resolution.split("x")
            w, h = int(w), int(h)
            latent_w, latent_h = w // 8, h // 8
            z = torch.randn([1, 4, latent_h, latent_w], device="cuda").bfloat16()
            z = z.repeat(2, 1, 1, 1)

            cap_tok = tokenizer.encode(caption, truncation=False)
            null_cap_tok = tokenizer.encode("", truncation=False)
            tok = torch.zeros([2, max(len(cap_tok), len(null_cap_tok))], dtype=torch.long, device="cuda")
            tok_mask = torch.zeros_like(tok, dtype=torch.bool)
            tok[0, :len(cap_tok)] = torch.tensor(cap_tok)
            tok[1, :len(null_cap_tok)] = torch.tensor(null_cap_tok)
            tok_mask[0, :len(cap_tok)] = True
            tok_mask[1, :len(null_cap_tok)] = True

            with torch.no_grad():
                cap_feats = model_lm.get_decoder()(input_ids=tok).last_hidden_state
            model_kwargs = dict(cap_feats=cap_feats, cap_mask=tok_mask, cfg_scale=cfg_scale)

            if dist.get_rank() == 0:
                print(f"caption: {caption}")
                print(f"num_sampling_steps: {num_sampling_steps}")
                print(f"cfg_scale: {cfg_scale}")

            samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
            samples = samples[:1]

            factor = 0.18215 if train_args.vae != 'sdxl' else 0.13025
            print(f"vae factor: {factor}")
            samples = vae.decode(samples / factor).sample
            samples = (samples + 1.) / 2.
            samples.clamp_(0., 1.)
            img = to_pil_image(samples[0])

            if response_queue is not None:
                response_queue.put(img)

        except Exception:
            print(traceback.format_exc())
            response_queue.put(ModelFailure())


def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None,
                       choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma",
                       choices=["constant", "SBDM", "sigma", "linear", "decreasing",
                                "increasing-decreasing"],
                       help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean",
                       choices=[None, "Mean", "Tweedie", "Euler"],
                       help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04,
                       help="size of the last step taken")


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main():
    parser = argparse.ArgumentParser()

    # use python demo.py mode --kwargs to specify sampling mode to ODE or SDE
    # e.g. python demo.py ODE --ckpt ${ckpt_path}
    # ODE will be used if mode is not explicitly provided
    mode = sys.argv[1]
    if mode not in ["ODE", "SDE"]:
        mode = "ODE"

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ema", action="store_true")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    args = parser.parse_known_args()[0]

    if args.num_gpus != 1:
        raise NotImplementedError("Multi-GPU Inference is not yet supported")

    args.sampler_mode = mode

    master_port = find_free_port()

    processes = []
    request_queues = []
    response_queue = mp.Queue()
    mp_barrier = mp.Barrier(args.num_gpus + 1)
    for i in range(args.num_gpus):
        request_queues.append(mp.Queue())
        p = mp.Process(target=model_main,
                       args=(args, master_port, i, request_queues[i], response_queue if i == 0 else None, mp_barrier))
        p.start()
        processes.append(p)

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(
f"""# Lumina-T2I image generation demo

**Model path:** {os.path.abspath(args.ckpt)}
 
**ema**: {args.ema}"""
            )
        with gr.Row():
            with gr.Column():
                cap = gr.Textbox(lines=2, label="Caption", interactive=True)
                with gr.Row():
                    num_sampling_steps = gr.Slider(
                        minimum=1, maximum=1000, value=30, interactive=True,
                        label="Sampling steps"
                    )
                    cfg_scale = gr.Slider(
                        minimum=1., maximum=20., value=4., interactive=True,
                        label="CFG scale"
                    )
                    solver = gr.Dropdown(
                        value="euler",
                        choices=["euler", "dopri5", "dopri8"],
                        label = "solver"
                    )
                with gr.Row():
                    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
                    if train_args.image_size == 256:
                        res_choices = ["256x256", "128x512", "512x128"]
                    elif train_args.image_size == 512:
                        res_choices = ["512x512", "256x1024", "1024x256"]
                    elif train_args.image_size == 1024:
                        res_choices = ["1024x1024", "512x2048", "2048x512"]
                    else:
                        raise NotImplementedError

                    resolution = gr.Dropdown(
                        value=res_choices[0],
                        choices=res_choices
                    )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    reset_btn = gr.ClearButton([cap, num_sampling_steps, cfg_scale])
            with gr.Column():
                output_img = gr.Image(label="Generated image", interactive=False)

        def on_submit(*args):
            for q in request_queues:
                q.put(args)
            result = response_queue.get()
            if isinstance(result, ModelFailure):
                raise RuntimeError
            return result

        submit_btn.click(on_submit, [cap, resolution, num_sampling_steps, cfg_scale, solver], [output_img])

    mp_barrier.wait()
    demo.queue().launch(
        share=True, server_name="0.0.0.0",
    )

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
