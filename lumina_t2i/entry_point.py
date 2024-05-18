import os
import click
import warnings
import builtins

from .utils.group import DefaultGroup
from .utils.cli import main


def none_or_str(value):
    if value == "None":
        return None
    return value


def version(ctx, _, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo("1.0.0")
    ctx.exit()


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


global_options = [
    click.option(
        "--num_gpus", type=int, default=1, help="number of gpus you want to use."
    ),
    click.option(
        "--ckpt",
        type=str,
        default=None,
        help="pretrained Lumina-T2X model checkpoint path.",
    ),
    click.option(
        "--ckpt_lm",
        type=str,
        default=None,
        help="pretrained LLM model checkpoint path.",
    ),
    click.option("--ema", is_flag=True, help="whether to load ema model."),
    click.option(
        "--precision",
        type=click.Choice(["bf16", "fp32"]),
        default="bf16",
        help="precision of inference for model.",
    ),
    click.option(
        "-c",
        "--config",
        type=str,
        default="cofing/infer/settings.yaml",
        help="setting for inference with different parameter.",
    ),
    click.option(
        "--token",
        type=str,
        default=False,
        help="huggingface token for accessing gated model.",
    ),
]

transport_options = [
    click.option(
        "--path-type", type=click.Choice(["Linear", "GVP", "VP"]), default="Linear"
    ),
    click.option(
        "--prediction",
        type=click.Choice(["velocity", "score", "noise"]),
        default="velocity",
    ),
    click.option(
        "--loss-weight",
        type=click.Choice([None, "velocity", "likelihood"]),
        default=None,
    ),
    click.option("--sample-eps", type=float),
    click.option("--train-eps", type=float),
]

ode_options = [
    click.option("-a", "--atol", type=float, default=1e-6, help="Absolute tolerance"),
    click.option("-r", "--rtol", type=float, default=1e-3, help="Relative tolerance"),
    click.option("--reverse", is_flag=True, help=""),
    click.option("--likelihood", is_flag=True, help=""),
]

sde_options = [
    click.option(
        "--sampling-method", type=click.Choice(["Euler", "Heun"]), default="Euler"
    ),
    click.option(
        "--diffusion-form",
        type=click.Choice(
            [
                "constant",
                "SBDM",
                "sigma",
                "linear",
                "decreasing",
                "increasing-decreasing",
            ]
        ),
        default="sigma",
        help="form of diffusion coefficient in the SDE",
    ),
    click.option("--diffusion-norm", type=float, default=1.0),
    click.option(
        "--last-step",
        type=click.Choice([None, "Mean", "Tweedie", "Euler"]),
        default="Mean",
        help="form of last step taken in the SDE",
    ),
    click.option(
        "--last-step-size", type=float, default=0.04, help="size of the last step taken"
    ),
]

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(cls=DefaultGroup, context_settings=CONTEXT_SETTINGS, default="infer")
@click.option(
    "-v",
    "--version",
    is_flag=True,
    callback=version,
    expose_value=False,
    is_eager=True,
    help="Print version info.",
)
def entry_point():
    pass


@add_options(global_options)
@click.argument("output_path", type=str, default="./", required=False, nargs=1)
@click.argument("text", type=str, required=True, nargs=1)
@entry_point.command(default=True)
def infer(num_gpus, ckpt, ckpt_lm, ema, precision, config, token, text, output_path):
    main(num_gpus, ckpt, ckpt_lm, ema, precision, config, token, text, output_path)


@click.argument("output_dir", type=str, required=True, nargs=1)
@click.argument("torch_weight_path", type=str, required=True, nargs=1)
@entry_point.command()
def convert(torch_weight_path, output_dir):
    import torch
    from safetensors.torch import save_file

    file_path, ext = os.path.splitext(torch_weight_path)
    if ext != ".pth":
        raise ValueError("Only `.pth` models are supported for conversion.")
    file_name = file_path.split("/")[-1]
    output_path = os.path.join(output_dir, file_name + ".safetensors")

    print(f"Loading your current `.pth` model {torch_weight_path}")
    if os.path.exists(output_path):
        raise ValueError(
            f"The output directory contains a model file with the same name `{file_name}`. "
            f"Please check if the `output_dir` {output_dir} is correct."
        )
    torch_weight_dict = torch.load(torch_weight_path, map_location="cpu")

    print(f"Saving model with `safetensors` format at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    save_file(torch_weight_dict, output_path)
    print("Done.")
