import argparse
import builtins
import json
import multiprocessing as mp
import os
import socket
import traceback

import gradio as gr
from omegaconf import OmegaConf
import torch

from models.util import instantiate_from_config
from models.vocoder.bigvgan.models import VocoderBigVGAN


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
    def __init__(self, args, model, config, vocoder=None) -> None:
        self.args = args
        self.model = model
        self.channel_dim = self.model.channels
        self.config = config

    def gen_test_sample(self, prompt, steps, cfg_scale, solver):
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
                c = self.model.get_learned_conditioning(prompt["ori_caption"])

            if self.channel_dim > 0:
                shape = [
                    self.channel_dim,
                    self.args.H,
                    self.args.W,
                ]
            else:
                shape = [1, self.args.H, self.args.W]

            x0 = torch.randn(shape, device=self.model.device)

            if cfg_scale == 1:  # w/o cfg
                sample, _ = self.model.sample(
                    c,
                    1,
                    timesteps=steps,
                    x_latent=x0,
                    solver=solver,
                )
            else:  # cfg
                sample, _ = self.model.sample_cfg(
                    c,
                    cfg_scale,
                    uc,
                    1,
                    timesteps=steps,
                    x_latent=x0,
                    solver=solver,
                )

            x_samples_ddim = self.model.decode_first_stage(sample)

        return x_samples_ddim


@torch.no_grad()
def model_main(args, rank, request_queue, response_queue, mp_barrier):
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
    dtype = torch.float32

    if args.num_gpus > 1:
        raise NotImplementedError("Inference with >1 GPUs not yet supported")

    print(f"Creating Model: Lumina-T2A")
    # latent_size = train_args.image_size // 8
    config = OmegaConf.load(args.config_path)
    model = load_model_from_config(config, args.ckpt)
    model.eval().to(device)

    print(f"Creating Decoder: Vocoder")
    vocoder = VocoderBigVGAN(args.vocoder_ckpt, device)

    print("Creating Generator")
    generator = GenSamples(args, model, config, vocoder)

    if args.ema:
        print("Loading ema model.")

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
                    samples_ddim = generator.gen_test_sample(cap, num_sampling_steps, cfg_scale, solver)
                    samples_ddim = samples_ddim.squeeze(0).cpu().numpy()
                    wav = vocoder.vocode(samples_ddim)
                print("> generated audio, done.")

                if response_queue is not None:
                    response_queue.put(((args.sample_rate, wav), metadata))
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
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vocoder_ckpt", type=str, required=True)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--sample_rate", type=int, required=True, choices=[16000])
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
                i,
                request_queues[i],
                response_queue if i == 0 else None,
                mp_barrier,
            ),
        )
        p.start()
        processes.append(p)

    description = """
    # Lumina Text-to-Music

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
                        minimum=10,
                        maximum=100,
                        value=40,
                        step=1,
                        interactive=True,
                        label="Sampling steps",
                    )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=int(1e5),
                        value=100,
                        step=1,
                        interactive=True,
                        label="Seed (0 for random)",
                    )
                with gr.Accordion("Advanced Settings for Resolution Extrapolation", open=False):
                    with gr.Row():
                        solver = gr.Dropdown(
                            value="euler",
                            choices=["euler"],
                            label="solver",
                        )
                        cfg_scale = gr.Slider(
                            minimum=2.0,
                            maximum=5.0,
                            value=5.0,
                            interactive=True,
                            label="CFG scale",
                        )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column():
                output_aud = gr.Audio(
                    label="Generated music",
                    interactive=False,
                    format="wav",
                )
                with gr.Accordion(label="Generation Parameters", open=True):
                    gr_metadata = gr.JSON(label="metadata", show_label=False)

        with gr.Row():
            gr.Examples(
                [
                    [
                        "This country song with banjo, acoustic piano, violin, and upright bass is an upbeat, uptempo tune that will get your feet tapping."
                    ],
                    [
                        "The upbeat instrumental song features punchy digital drums, lively piano harmony, and funky bass line, accompanied by perky synthesised violins and various background sounds superimposed by music, creating a happy and lively atmosphere with a fast tempo and a sound of beeping."
                    ],
                    [
                        "A sentimental country song featuring gentle fiddle and acoustic guitar alongside electric guitar, with a slow tempo and easygoing melody that evokes a sense of longing."
                    ],
                    [
                        "This vintage guitar demonstration on YouTube showcases a passionate and spirited soloist's unique and melancholic country music style, despite the buzzing noise, bad audio quality, and ambient room noise."
                    ],
                    [
                        "The song is characterized by a cheerful instrumental with a dominant marimba melody, but suffers from excessive noise and low-quality production."
                    ],
                    [
                        "This is a suspenseful and mysterious sci-fi instrumental music with a booming lower harmony, string section harmony, horn section, and tuba that builds tension as if exploring a dangerous territory."
                    ],
                    [
                        "A bagpipe ensemble performs a quick melody with unison and drone accompaniment, featuring trills and ornamental flourishes."
                    ],
                    [
                        "A fast-paced, intense rock song with amplified electric guitar and stringed instruments showcasing skilled and dexterous musicianship, featuring sonic power and vibrato, recorded at home."
                    ],
                    [
                        "A joyful duet with a medium tempo, featuring a charming accordion melody, tambourine beats, and hand percussion alongside keyboard harmony, delivering a simple and cheerful Christmas song with jingle bells in Spanish and animated, merry lyrics tailored towards children."
                    ],
                    [
                        "This instrumental progressive rock song features a complex, electric guitar solo with impressive tapping techniques."
                    ],
                    [
                        "A low-quality classical piece with a wide strings, low brass, and woodwinds melody that creates a suspenseful and intense atmosphere."
                    ],
                    [
                        "This bluesy song features a soothing combination of harmonica, acoustic guitar arpeggio, and pad synth."
                    ],
                    ["The song features a rapid and complex brass motif led by a high pitched trumpet."],
                    [
                        "A poorly produced, noisy and mono cover song with an aggressive yet groovy funky acoustic bass guitar solo melody."
                    ],
                    [
                        "This folk music song has an uplifting and medium-to-uptempo rhythm highlighted by the sounds of the violin, horn, brass, xylophone, and drums."
                    ],
                    [
                        "An amateur recording featuring two acoustic guitars with a technically challenging and flamenco-inspired nylon string guitar played in a somewhat sloppy manner."
                    ],
                    [
                        "A mystical, solo instrumental piece featuring the tranquil sounds of the bass flute creates an enchanting fantasy forest ambiance with a touch of reverb."
                    ],
                    [
                        "The instrumental track features a relaxing mix of guitar and piano music set to a moderate tempo with lingering bassline, pads, and synth sounds to create a meditative listening experience."
                    ],
                    [
                        "The song features a passionate electric guitar solo melody with a chorus pedal effect, but is marred by low quality, noise, and lacking stereo presence."
                    ],
                    ["The haunting melody of the cello evokes a deep sense of emotion in this classical masterpiece."],
                ],
                [cap],
                label="Examples",
                examples_per_page=10,
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
    demo.queue().launch(server_name="0.0.0.0", server_port=7865)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
