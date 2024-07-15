import os
from pathlib import Path

import torch
from loguru import logger

from hydit.config import get_args
import lumina_next_t2i_mini.models as models

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import polygraphy.backend.onnx.loader

import models


def _to_tuple(val):
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            val = [val[0], val[0]]
        elif len(val) == 2:
            val = tuple(val)
        else:
            raise ValueError(f"Invalid value: {val}")
    elif isinstance(val, (int, float)):
        val = (val, val)
    else:
        raise ValueError(f"Invalid value: {val}")
    return val


class ExportONNX(object):
    def __init__(self, args, models_root_path):
        self.args = args
        self.model = None
        # Set device and disable gradient
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)

        # Check arguments
        self.root = models_root_path
        logger.info(f"Got text-to-image model root path: {models_root_path}")

        # Create folder to save onnx model
        self.onnx_workdir = self.args.onnx_workdir
        self.onnx_export = os.path.join(self.onnx_workdir, "export/model.onnx")
        os.makedirs(os.path.join(self.onnx_workdir, "export"), exist_ok=True)
        self.onnx_modify = os.path.join(self.onnx_workdir, "export_modified/model.onnx")
        os.makedirs(os.path.join(self.onnx_workdir, "export_modified"), exist_ok=True)
        self.onnx_fmha = os.path.join(self.onnx_workdir, "exxport_modified_fmha/model.onnx")
        os.makedirs(os.path.join(self.onnx_workdir, "export_modified_fmha"), exist_ok=True)

    def load_model(self):
        # ========================================================================
        # Create model structure and load the checkpoint
        logger.info(f"Building Lumina-NextDiT model...")
        
        logger.info("Loading model_args.pth for the Lumina-NextDiT model.")
        args_path = os.path.join(args.ckpt, "model_args.pth")
        if not os.path.exists(args_path):
            raise ValueError(f"model_args not exists: {model_path}")
        train_args = torch.load(args_path)
        
        image_size = _to_tuple(self.args.image_size)
        latent_size = (image_size[0] // 8, image_size[1] // 8)

        model_dir = self.root / "model"
        model_path = model_dir / f"pytorch_model_{self.args.load_key}.pt"
        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")

        # Build model structure
        self.model = models.__dict__[train_args.model](
            qk_norm=train_args.qk_norm,
            cap_feat_dim=2048,
            use_flash_attn=args.use_flash_attn,
        )
        # Load model checkpoint
        logger.info(f"Loading torch model {model_path}...")
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device, dtype=torch.bfloat16)
        logger.info(f"Loading torch model finished")
        logger.info("==================================================")
        logger.info(f"                Model is ready.                  ")
        logger.info("==================================================")

    def export(self):
        if self.model is None:
            self.load_model()

        # Construct model inputs
        latent_model_input = torch.randn(2, 4, 128, 128, device=self.device).half()
        t_expand = torch.randint(0, 1000, [2], device=self.device).half()
        prompt_embeds = torch.randn(2, 256, 2048, device=self.device).half()
        attention_mask = torch.randint(0, 2, [2, 256], device=self.device).long()

        save_to = self.onnx_export
        logger.info(f"Exporting ONNX model {save_to}...")
        logger.info(f"Exporting ONNX external data {save_to.parent}...")
        model_args = (
            latent_model_input,
            t_expand,
            prompt_embeds,
            attention_mask
        )
        torch.onnx.export(self.model,
                          model_args,
                          str(save_to),
                          export_params=True,
                          opset_version=17,
                          do_constant_folding=True,
                          input_names=["x", "t", "cap_feats", "cap_mask"],
                          output_names=["output"],
                          dynamic_axes={"x": {0: "2B", 2: "H", 3: "W"}, "t": {0: "2B"},
                                        "cap_feats": {0: "2B"},
                                        "cap_mask": {0: "2B"}},
                          )
        logger.info("Exporting onnx finished")

    def postprocessing(self):
        load_from = self.onnx_export
        save_to = self.onnx_modify
        logger.info(f"Postprocessing ONNX model {load_from}...")

        onnxModel = onnx.load(str(load_from), load_external_data=False)
        onnx.load_external_data_for_model(onnxModel, str(load_from.parent))
        graph = gs.import_onnx(onnxModel)

        # ADD GAMMA BETA FOR LN
        for node in graph.nodes:
            if node.name == "/final_layer/norm_final/LayerNormalization":
                constantKernel = gs.Constant("final_layer.norm_final.weight",
                                             np.ascontiguousarray(np.ones((1408,), dtype=np.float16)))
                constantBias = gs.Constant("final_layer.norm_final.bias",
                                           np.ascontiguousarray(np.zeros((1408,), dtype=np.float16)))
                node.inputs = [node.inputs[0], constantKernel, constantBias]

        graph.cleanup().toposort()
        onnx.save(gs.export_onnx(graph.cleanup()),
                  str(save_to),
                  save_as_external_data=True,
                  all_tensors_to_one_file=False,
                  location=str(save_to.parent),
                  )
        logger.info(f"Postprocessing ONNX model finished: {save_to}")

    def fuse_attn(self):
        load_from = self.onnx_modify
        save_to = self.onnx_fmha
        logger.info(f"FuseAttn ONNX model {load_from}...")

        onnx_graph = polygraphy.backend.onnx.loader.fold_constants(
            onnx.load(str(load_from)),
            allow_onnxruntime_shape_inference=True,
        )
        graph = gs.import_onnx(onnx_graph)

        cnt = 0
        for node in graph.nodes:

            if node.op == "Softmax" and node.i().op == "MatMul" and node.o().op == "MatMul" and \
                    node.o().o().op == "Transpose":

                if "pooler" in node.name:
                    continue

                if "attn1" in node.name:
                    matmul_0 = node.i()
                    transpose = matmul_0.i(1, 0)
                    transpose.attrs["perm"] = [0, 2, 1, 3]
                    k = transpose.outputs[0]
                    q = gs.Variable("transpose_0_v_{}".format(cnt), np.dtype(np.float16))
                    transpose_0 = gs.Node("Transpose", "Transpose_0_{}".format(cnt),
                                          attrs={"perm": [0, 2, 1, 3]},
                                          inputs=[matmul_0.inputs[0]],
                                          outputs=[q])
                    graph.nodes.append(transpose_0)

                    matmul_1 = node.o()
                    v = gs.Variable("transpose_1_v_{}".format(cnt), np.dtype(np.float16))
                    transpose_1 = gs.Node("Transpose", "Transpose_1_{}".format(cnt),
                                          attrs={"perm": [0, 2, 1, 3]},
                                          inputs=[matmul_1.inputs[1]],
                                          outputs=[v])
                    graph.nodes.append(transpose_1)

                    output_variable = node.o().o().outputs[0]
                    # fMHA_v = gs.Variable("fMHA_v", np.dtype(np.float16))
                    fMHA = gs.Node("fMHAPlugin", "fMHAPlugin_1_{}".format(cnt),
                                   # attrs={"scale": 1.0},
                                   inputs=[q, k, v],
                                   outputs=[output_variable])
                    graph.nodes.append(fMHA)
                    node.o().o().outputs = []
                    cnt = cnt + 1

                elif "attn2" in node.name:
                    matmul_0 = node.i()
                    transpose_q = matmul_0.i()
                    transpose_k = matmul_0.i(1, 0)
                    matmul_1 = node.o()
                    transpose_v = matmul_1.i(1, 0)
                    q = transpose_q.inputs[0]
                    k = transpose_k.inputs[0]
                    v = transpose_v.inputs[0]
                    output_variable = node.o().o().outputs[0]
                    fMHA = gs.Node("fMHAPlugin", "fMHAPlugin_1_{}".format(cnt),
                                   # attrs={"scale": 1.0},
                                   inputs=[q, k, v],
                                   outputs=[output_variable])
                    graph.nodes.append(fMHA)
                    node.o().o().outputs = []
                    cnt = cnt + 1

        logger.info("mha count: ", cnt)

        onnx.save(gs.export_onnx(graph.cleanup()),
                  str(save_to),
                  save_as_external_data=True,
                  )
        logger.info(f"FuseAttn ONNX model finished: {save_to}")


if __name__ == "__main__":
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    exporter = ExportONNX(args, models_root_path)
    exporter.export()
    exporter.postprocessing()
    # exporter.fuse_attn()