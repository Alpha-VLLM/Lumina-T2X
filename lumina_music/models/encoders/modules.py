from importlib_resources import files
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from models.util import count_params
from .CLAP.clap import TextEncoder
from .CLAP.utils import read_config_as_args


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenFLANEmbedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self, version="google/flan-t5-large", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length  # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)  # tango的flanT5是不定长度的batch，这里做成定长的batch
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLAPEmbedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text from microsoft"""

    def __init__(self, weights_path, freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()

        model_state_dict = torch.load(weights_path, map_location=torch.device("cpu"))["model"]
        match_params = dict()
        for key in list(model_state_dict.keys()):
            if "caption_encoder" in key:
                match_params[key.replace("caption_encoder.", "")] = model_state_dict[key]

        config_as_str = files("models").joinpath("encoders/CLAP/config.yml").read_text()
        args = read_config_as_args(config_as_str, is_config_str=True)

        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model)  # args.text_model
        self.caption_encoder = TextEncoder(args.d_proj, args.text_model, args.transformer_embed_dim)

        self.max_length = max_length
        self.device = device

        if freeze:
            self.freeze()

        print(
            f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params."
        )

    def freeze(self):  # only freeze
        self.caption_encoder.base = self.caption_encoder.base.eval()
        for param in self.caption_encoder.base.parameters():
            param.requires_grad = False

    def encode(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)

        outputs = self.caption_encoder.base(input_ids=tokens)
        z = self.caption_encoder.projection(outputs.last_hidden_state)
        return z


class FrozenCLAPFLANEmbedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text from microsoft"""

    def __init__(
        self, weights_path, t5version="google/t5-v1_1-large", freeze=True, device="cuda", max_length=77
    ):  # clip-vit-base-patch32
        super().__init__()

        model_state_dict = torch.load(weights_path, map_location=torch.device("cpu"))["model"]
        match_params = dict()
        for key in list(model_state_dict.keys()):
            if "caption_encoder" in key:
                match_params[key.replace("caption_encoder.", "")] = model_state_dict[key]

        config_as_str = files("ldm").joinpath("modules/encoders/CLAP/config.yml").read_text()
        args = read_config_as_args(config_as_str, is_config_str=True)

        # if os.path.exists('/apdcephfs/share_1316500/nlphuang/results/Text_to_audio/pretrained/CLAP'):
        #     root = '/apdcephfs'
        # else:
        #     root = '/apdcephfs_intern'

        self.clap_tokenizer = AutoTokenizer.from_pretrained(args.text_model)  # args.text_model
        self.caption_encoder = TextEncoder(args.d_proj, args.text_model, args.transformer_embed_dim)

        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5version)
        self.t5_transformer = T5EncoderModel.from_pretrained(t5version)

        self.max_length = max_length
        self.to(device=device)
        if freeze:
            self.freeze()

        print(
            f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params."
        )

    def freeze(self):
        self.caption_encoder = self.caption_encoder.eval()
        for param in self.caption_encoder.parameters():
            param.requires_grad = False

    def to(self, device):
        self.t5_transformer.to(device)
        self.caption_encoder.to(device)
        self.device = device

    def encode(self, text):
        ori_caption = text["ori_caption"]
        struct_caption = text["struct_caption"]
        # print(ori_caption,struct_caption)
        clap_batch_encoding = self.clap_tokenizer(
            ori_caption,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        ori_tokens = clap_batch_encoding["input_ids"].to(self.device)
        t5_batch_encoding = self.t5_tokenizer(
            struct_caption,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        struct_tokens = t5_batch_encoding["input_ids"].to(self.device)
        outputs = self.caption_encoder.base(input_ids=ori_tokens)
        z = self.caption_encoder.projection(outputs.last_hidden_state)
        z2 = self.t5_transformer(input_ids=struct_tokens).last_hidden_state
        return torch.concat([z, z2], dim=1)
