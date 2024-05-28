import os

import click

from .utils.cli import main
from .utils.group import DefaultGroup


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
    click.option("--num_gpus", type=int, default=1, help="number of gpus you want to use."),
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
    click.option("--path-type", type=click.Choice(["Linear", "GVP", "VP"]), default="Linear"),
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
@click.argument("weight_path", type=str, required=True, nargs=1)
@entry_point.command()
def convert(weight_path, output_dir):
    """
    convert torch model weight `.pth` into `.safetensors`

    Args:
        weight_path (str): pytorch model path
        output_dir (str): saved directory, supports saving files with different names in the same directory.

    """
    from safetensors.torch import load_file, save_file
    import torch

    supported_model_type = (".pth", ".safetensors")

    file_path, ext = os.path.splitext(weight_path)
    if ext != ".pth" and ext != ".safetensors":
        raise ValueError(f"Only {supported_model_type} models are supported for conversion.")

    file_name = file_path.split("/")[-1]
    print(f"Loading your current `{ext}` model {weight_path}")
    os.makedirs(output_dir, exist_ok=True)

    if ext == supported_model_type[0]:
        target_ext = supported_model_type[1]
        output_path = os.path.join(output_dir, file_name + target_ext)

        torch_weight_dict = torch.load(weight_path, map_location="cpu")
        save_file(torch_weight_dict, output_path)
        print(f"Saving model with `{supported_model_type[1]}` format at {output_dir}")

    elif ext == supported_model_type[1]:
        target_ext = supported_model_type[0]
        output_path = os.path.join(output_dir, file_name + target_ext)

        safetensors_weight_dict = load_file(weight_path, device="cpu")
        torch.save(safetensors_weight_dict, output_path)
        print(f"Saving model with `{ext}` format at {output_dir}")

    print("Done.")
