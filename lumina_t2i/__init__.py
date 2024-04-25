from .entry_point import *

# for cli setup

# import argparse
# import builtins
# import json
# import multiprocessing as mp
# import os
# import socket
# import traceback

# import fairscale.nn.model_parallel.initialize as fs_init
# import gradio as gr
# import torch
# import torch.distributed as dist
# from torchvision.transforms.functional import to_pil_image

# from models import *
# from .transport import create_transport, Sampler

# def setup_dist():
#     os.environ["MASTER_PORT"] = str(master_port)
#     os.environ["MASTER_ADDR"] = "127.0.0.1"
#     os.environ["RANK"] = str(rank)
#     os.environ["WORLD_SIZE"] = str(args.num_gpus)

#     dist.init_process_group("nccl")
#     # set up fairscale environment because some methods of the Lumina model need it,
#     # though for single-GPU inference fairscale actually has no effect
#     fs_init.initialize_model_parallel(args.num_gpus)
#     torch.cuda.set_device(rank)

