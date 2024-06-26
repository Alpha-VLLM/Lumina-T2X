import argparse
import builtins
import json
import multiprocessing as mp
import os
import random
import socket
import traceback

import gradio as gr
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from tqdm import tqdm

from models.util import instantiate_from_config
from models.vocoder.bigvgan.models import VocoderBigVGAN
import n2s_openai as n2s


def load_model_from_config(config, ckpt=None, verbose=True):
    model = instantiate_from_config(config.model)
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]

        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    else:
        raise ValueError("Note: no ckpt is loaded !!!")

    return model


class ModelFailure:
    pass


class GenSamples:
    def __init__(self, args, model, outpath, config, vocoder=None) -> None:
        self.args = args
        self.model = model
        self.outpath = outpath
        self.vocoder = vocoder
        self.channel_dim = self.model.channels
        self.config = config

    def gen_test_sample(self, prompt, steps, cfg_scale, solver):
        # prompt is {'ori_caption':’xxx‘,'struct_caption':'xxx'}
        uc = None
        if cfg_scale != 1.0:
            try:
                uc = self.model.get_learned_conditioning({"ori_caption": "", "struct_caption": ""})
            except:
                uc = self.model.get_learned_conditioning(prompt)
        for n in range(self.args.n_iter):

            try:
                c = self.model.get_learned_conditioning(prompt)
            except:
                c = self.model.get_learned_conditioning(prompt)

            if self.channel_dim > 0:
                shape = [
                    self.channel_dim,
                    self.args.H,
                    self.args.W,
                ]
            else:
                shape = [1, self.args.H, self.args.W]

            x0 = torch.randn(shape, device=self.model.device)

            if cfg_scale == 1:
                sample, _ = self.model.sample(c, 1, timesteps=steps, x_latent=x0, solver=solver)
            else:
                sample, _ = self.model.sample_cfg(c, cfg_scale, uc, 1, timesteps=steps, x_latent=x0, solver=solver)

            x_samples_ddim = self.model.decode_first_stage(sample)

        return x_samples_ddim


@torch.no_grad()
def model_main(args, master_port, rank, request_queue, response_queue, mp_barrier):
    # override the default print function since the delay can be large for child process
    original_print = builtins.print

    # Redefine the print function with flush=True by default
    def print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        original_print(*args, **kwargs)

    # Override the built-in print with the new version
    builtins.print = print

    os.environ["RANK"] = str(rank)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = {"fp32": torch.float32}[args.precision]

    if args.num_gpus > 1:
        raise NotImplementedError("Inference with >1 GPUs not yet supported")

    print(f"Creating Model: Lumina-T2Audio")
    config = OmegaConf.load(args.config_path)
    model = load_model_from_config(config, args.ckpt)
    model.eval().to(device)

    print(f"Creating Decoder: Vocoder")
    vocoder = VocoderBigVGAN(args.vocoder_ckpt, device)

    print("Creating Generator")
    generator = GenSamples(args, model, config, vocoder)

    mp_barrier.wait()

    with torch.autocast("cuda", dtype):
        while True:
            (
                cap,
                num_sampling_steps,
                cfg_scale,
                solver,
                seed,
            ) = request_queue.get()

            metadata = dict(
                cap=cap,
                num_sampling_steps=num_sampling_steps,
                cfg_scale=cfg_scale,
                solver=solver,
                seed=seed,
            )
            print("> params:", json.dumps(metadata, indent=2))

            try:
                with model.ema_scope():
                    struct_caption = n2s.get_struct(cap)
                    prompt = {"ori_caption": [cap], "struct_caption": [struct_caption]}
                    print(f"The structed caption by Chatgpt is : {struct_caption}")
                    samples_ddim = generator.gen_test_sample(prompt, num_sampling_steps, cfg_scale, solver)
                    samples_ddim = samples_ddim.squeeze(0).cpu().numpy()
                    wav = vocoder.vocode(samples_ddim)
                print("> generated audio, done.")

                if response_queue is not None:
                    response_queue.put(((args.sample_rate, wav), metadata))

            except Exception:
                print(traceback.format_exc())
                response_queue.put(ModelFailure())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vocoder_ckpt", type=str, required=True)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--save_wav", action="store_true")
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--precision", default="fp32", choices=["bf16", "fp32"])
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=20,
        help="latent height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=312,
        help="latent width, in pixel space",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="huggingface read token for accessing gated repo.",
    )

    args = parser.parse_known_args()[0]

    if args.num_gpus != 1:
        raise NotImplementedError("Multi-GPU Inference is not yet supported")

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
                60001,
                i,
                request_queues[i],
                response_queue if i == 0 else None,
                mp_barrier,
            ),
        )
        p.start()
        processes.append(p)

    description = """
    # Lumina Text-to-Audio

    ### Lumina Text-to-Audio requires using `gpt-3.5-turbo` to generate structure caption.

    ### <span style="color: red;">(Please ensure that the OpenAI API key is set in lines 8 and 9 of the `n2s_openai.py` file. If using other proxies, please set the `base_url` accordingly.)

    ### Before using it, please set your `OpenAI API key` to ensure correct generation of structured descriptions and suitable audio.

    ### <span style="color: red;">We will soon release a version that does not require structure caption.

    """
    with gr.Blocks(css="./style.css") as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                cap = gr.Textbox(
                    lines=2,
                    label="Caption",
                    interactive=True,
                    value="",
                    placeholder="Enter a caption.",
                )
                with gr.Row():
                    num_sampling_steps = gr.Slider(
                        minimum=20,
                        maximum=200,
                        value=100,
                        step=1,
                        interactive=True,
                        label="Sampling steps",
                    )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=int(1e5),
                        value=42,
                        step=1,
                        interactive=True,
                        label="Seed (0 for random)",
                    )
                with gr.Accordion("Advanced Settings for Resolution Extrapolation", open=False):
                    with gr.Row():
                        solver = gr.Dropdown(
                            value="euler",
                            choices=["euler"],
                            label="Solver",
                        )
                        cfg_scale = gr.Slider(
                            minimum=2.0,
                            maximum=5.0,
                            value=3.0,
                            interactive=True,
                            label="CFG scale",
                        )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column():
                output_aud = gr.Audio(
                    label="Generated audio",
                    interactive=False,
                    format="wav",
                )
                with gr.Accordion(label="Generation Parameters", open=True):
                    gr_metadata = gr.JSON(label="metadata", show_label=False)

        with gr.Row():
            gr.Examples(
                [
                    ["A honking horn from an oncoming train"],
                    ["A quiet whirring of an airplane"],
                    ["Rapid typing on a keyboard"],
                    ["A cup is filled from a faucet"],
                    ["A muffled helicopter engine flying"],
                    ["Continuous rain falling on surface"],
                    ["A toy helicopter flying then powering down before powering up"],
                    ["A door slamming shut"],
                    ["A bird chirps and water splashes lightly"],
                    ["An emergency vehicle sounds siren followed by a vehicle engine idling"],
                    ["A jackhammer drilling and vibrating continuously"],
                    ["Sizzling of food frying"],
                    ["White noise followed by clanking and then a toilet flushing"],
                    ["A person snoring"],
                    [
                        "A train running on railroad tracks drives by as a train horn blows several times alongside a railroad crossing signal ringing"
                    ],
                    ["A strong wind is blowing and constant background waves can be heard"],
                    ["A large bell rings out multiple times"],
                    ["A car is passing by with leaves rustling"],
                    ["Some humming followed by a toilet flushing"],
                    ["Tapping noise followed by splashing and gurgling water"],
                ],
                [cap],
                label="Examples",
            )

        def on_submit(*args):
            for q in request_queues:
                q.put(args)
            result = response_queue.get()
            audio, metadata = result

            if isinstance(result, ModelFailure):
                raise RuntimeError

            return audio, metadata

        submit_btn.click(
            on_submit,
            [
                cap,
                num_sampling_steps,
                cfg_scale,
                solver,
                seed,
            ],
            [output_aud, gr_metadata],
        )

    mp_barrier.wait()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
