# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import functools
import logging
import math
from typing import Optional, Tuple, List

# from apex.normalization import FusedRMSNorm as RMSNorm
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear, RowParallelLinear, ParallelEmbedding,
)
from flash_attn import flash_attn_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class PFRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.

        """
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#############################################################################
#             Embedding Layers for Timesteps and Class Labels               #
#############################################################################

class ParallelTimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                frequency_embedding_size, hidden_size, bias=True,
                gather_output=False,
                init_method=functools.partial(nn.init.normal_, std=0.02),
            ),
            nn.SiLU(),
            RowParallelLinear(
                hidden_size, hidden_size, bias=True, input_is_parallel=True,
                init_method=functools.partial(nn.init.normal_, std=0.02),
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32
            ) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([
                embedding, torch.zeros_like(embedding[:, :1])
            ], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class ParallelLabelEmbedder(nn.Module):
    r"""Embeds class labels into vector representations. Also handles label
    dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = ParallelEmbedding(
            num_classes + use_cfg_embedding, hidden_size,
            init_method=functools.partial(nn.init.normal_, std=0.02),
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(
                labels.shape[0], device=labels.device
            ) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            dist.broadcast(
                drop_ids,
                fs_init.get_model_parallel_src_rank(),
                fs_init.get_model_parallel_group(),
            )
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#############################################################################
#                               Core DiT Model                              #
#############################################################################


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int], qk_norm: bool):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = ColumnParallelLinear(
            dim, n_heads * self.head_dim, bias=False, gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )
        self.wk = ColumnParallelLinear(
            dim, self.n_kv_heads * self.head_dim, bias=False,
            gather_output=False, init_method=nn.init.xavier_uniform_,
        )
        self.wv = ColumnParallelLinear(
            dim, self.n_kv_heads * self.head_dim, bias=False,
            gather_output=False, init_method=nn.init.xavier_uniform_,
        )
        self.wo = RowParallelLinear(
            n_heads * self.head_dim, dim, bias=False,
            input_is_parallel=True, init_method=nn.init.xavier_uniform_,
        )

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_local_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_local_kv_heads * self.head_dim)
            # self.q_norm = nn.LayerNorm(self.n_local_heads * self.head_dim, elementwise_affine=False)
            # self.k_norm = nn.LayerNorm(self.n_local_kv_heads * self.head_dim, elementwise_affine=False)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as
        the target tensor 'x' for the purpose of broadcasting the frequency
        tensor during element-wise operations.

        Args:
            freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
            x (torch.Tensor): Target tensor for broadcasting compatibility.

        Returns:
            torch.Tensor: Reshaped frequency tensor.

        Raises:
            AssertionError: If the frequency tensor doesn't match the expected
                shape.
            AssertionError: If the target tensor 'x' doesn't have the expected
                number of dimensions.
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1
                 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency
        tensor.

        This function applies rotary embeddings to the given query 'xq' and
        key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
        input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors
        contain rotary embeddings and are returned as real tensors.

        Args:
            xq (torch.Tensor): Query tensor to apply rotary embeddings.
            xk (torch.Tensor): Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
                exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
                and key tensor with rotary embeddings.
        """
        with torch.cuda.amp.autocast(enabled=False):
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = Attention.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        if dtype in [torch.float16, torch.bfloat16]:
            output = flash_attn_func(xq, xk, xv, dropout_p=0., causal=False)
        else:
            n_rep = self.n_local_heads // self.n_local_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                xk.permute(0, 2, 1, 3),
                xv.permute(0, 2, 1, 3),
                dropout_p=0., is_causal=False,
            ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first
                layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third
                layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True,
            init_method=nn.init.xavier_uniform_,
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )

    @torch.compile
    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_experts_per_tok: int):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs: torch.Tensor):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.gate(inputs_squashed)
        # gate_logits = self.gate(cond).repeat(1, inputs.shape[1]).view(inputs_squashed.shape[0], -1)
        weights, selected_experts = torch.topk(
            gate_logits, self.num_experts_per_tok
        )
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(inputs)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs_squashed[batch_idx]
            )
        return results.view_as(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int,
                 multiple_of: int, ffn_dim_multiplier: float, norm_eps: float,
                 qk_norm: bool) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int):
            ffn_dim_multiplier (float):
            norm_eps (float):

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = FeedForward(
            dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                min(dim, 1024), 6 * dim, bias=True, gather_output=True,
                init_method=nn.init.zeros_,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)

            h = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa),
                freqs_cis,
            )
            out = h + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(h), shift_mlp, scale_mlp),
            )

        else:
            h = x + self.attention(
                self.attention_norm(x), freqs_cis,
            )
            out = h + self.feed_forward(self.ffn_norm(h))

        return out


class TransformerBlockSandwichNorm(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int,
                 multiple_of: int, ffn_dim_multiplier: float, norm_eps: float,
                 qk_norm: bool) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value in the FeedForward block.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension in the FeedForward block. Defaults to None.
            norm_eps (float): A small value added to the norm layer
                denominators to avoid division-by-zero.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = FeedForward(
            dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                min(dim, 1024), 6 * dim, bias=True, gather_output=True,
                init_method=nn.init.zeros_,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)

            h = x + self.attention_norm(gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm1(x), shift_msa, scale_msa),
                freqs_cis,
            ))
            out = h + self.ffn_norm(gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm1(h), shift_mlp, scale_mlp),
            ))

        else:
            h = x + self.attention_norm(self.attention(
                self.attention_norm1(x), freqs_cis,
            ))
            out = h + self.ffn_norm(self.feed_forward(self.ffn_norm1(h)))

        return out


class TransformerBlockSandwichNorm2(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int,
                 multiple_of: int, ffn_dim_multiplier: float, norm_eps: float,
                 qk_norm: bool, num_experts: int = 8, num_experts_per_tok: int = 2) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value in the FeedForward block.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension in the FeedForward block. Defaults to None.
            norm_eps (float): A small value added to the norm layer
                denominators to avoid division-by-zero.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm)
        self.num_experts = num_experts
        self.feed_forward: nn.Module
        if num_experts > 0:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(
                    dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,) for _ in range(num_experts)],
                gate=nn.Linear(dim, num_experts, bias=False), 
                num_experts_per_tok=num_experts_per_tok)
        else:
            self.feed_forward = FeedForward(
                dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention_norm1 = PFRMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = PFRMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                min(dim, 1024), 4 * dim, bias=True, gather_output=True,
                init_method=nn.init.zeros_,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        time_input: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if adaln_input is not None:
            scale_msa, gate_msa, scale_mlp, gate_mlp = \
                self.adaLN_modulation(adaln_input).chunk(4, dim=1)
            h = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm(self.attention(
                self.attention_norm1(x) * (1 + scale_msa.unsqueeze(1)),
                freqs_cis,
            ))
            if self.num_experts > 0:
                out = h + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm(self.feed_forward(
                    self.ffn_norm1(h) * (1 + scale_mlp.unsqueeze(1)),
                ))
            else:
                out = h + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm(self.feed_forward(
                    self.ffn_norm1(h) * (1 + scale_mlp.unsqueeze(1)),
                ))

        else:
            # h = x + self.attention_norm(self.attention(
            #     self.attention_norm1(x), freqs_cis,
            # ))
            # out = h + self.ffn_norm(self.feed_forward(self.ffn_norm1(h)))
            raise NotImplementedError

        return out


class ParallelFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.linear = ColumnParallelLinear(
            hidden_size, patch_size * patch_size * out_channels, bias=True,
            init_method=nn.init.zeros_, gather_output=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                min(hidden_size, 1024), 2 * hidden_size, bias=True,
                init_method=nn.init.zeros_, gather_output=True,
            ),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Llama(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.patch_size = patch_size

        self.x_embedder = ColumnParallelLinear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
            gather_output=True,
            init_method=nn.init.xavier_uniform_,
        )
        nn.init.constant_(self.x_embedder.bias, 0.)

        self.t_embedder = ParallelTimestepEmbedder(min(dim, 1024))
        self.y_embedder = ParallelLabelEmbedder(num_classes, min(dim, 1024),
                                                class_dropout_prob)

        self.layers = nn.ModuleList([
            TransformerBlockSandwichNorm2(layer_id, dim, n_heads, n_kv_heads, multiple_of,
                             ffn_dim_multiplier, norm_eps, qk_norm)
            for layer_id in range(n_layers)
        ])
        self.final_layer = ParallelFinalLayer(dim, patch_size, self.out_channels)

        assert (dim // n_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"
        self.freqs_cis = DiT_Llama.precompute_freqs_cis(
            dim // n_heads, 384,
            # theta=100.
        )

        self.dim = dim
        self.n_heads = n_heads

    def unpatchify(self, x: torch.Tensor, H, W) -> torch.Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        B = x.shape[0]
        C = self.out_channels
        P = self.patch_size

        x = x.reshape(shape=(B, H, W, P, P, C))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(B, C, H * P, W * P))
        return imgs

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        # assert (H, W) == (self.input_size, self.input_size)
        pH = pW = self.patch_size
        x = x.view(B, C, H // pH, pH, W // pW, pW)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)

        return x, H // pH, W // pW, self.freqs_cis[:H//pH, :W//pW].flatten(0,1).unsqueeze(0)

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent
           representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        x, H, W, freqs_cis = self.patchify(x)
        freqs_cis = freqs_cis.to(x.device)
        x = self.x_embedder(x)

        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        adaln_input = t + y

        for layer in self.layers:
            x = layer(
                x, freqs_cis,
                adaln_input=adaln_input,
                time_input=t,
            )

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, H, W)         # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, rope_scaling_factor=None, ntk_factor=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass
        for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

        if rope_scaling_factor is not None or ntk_factor is not None:
            assert rope_scaling_factor is not None and ntk_factor is not None
            # print("override freqs_cis", flush=True)
            self.freqs_cis = DiT_Llama.precompute_freqs_cis(
                self.dim // self.n_heads, 384, rope_scaling_factor=rope_scaling_factor, ntk_factor=ntk_factor
            )
        else:
            pass
            # print("use original freqs_cis", flush=True)
        
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """

        theta = theta * ntk_factor

        logger.info(f"theta {theta} rope scaling {rope_scaling_factor} ntk {ntk_factor}")
        freqs = 1.0 / (theta ** (
            torch.arange(0, dim, 4)[: (dim // 4)].float() / dim
        ))
        t = torch.arange(end, device=freqs.device, dtype=torch.float)  # type: ignore
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

        freqs_cis_h = freqs_cis.view(end, 1, dim//4, 1).repeat(1, end, 1, 1)
        freqs_cis_w = freqs_cis.view(1, end, dim//4, 1).repeat(end, 1, 1, 1)
        freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1).flatten(2)
        return freqs_cis

    def parameter_count(self) -> int:
        tensor_parallel_module_list = (
            ColumnParallelLinear, RowParallelLinear, ParallelEmbedding,
        )
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            is_tp_module = isinstance(module, tensor_parallel_module_list)
            for param in module.parameters(recurse=False):
                total_params += param.numel() * (
                    fs_init.get_model_parallel_world_size()
                    if is_tp_module else 1
                )
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


#############################################################################
#                                 DiT Configs                               #
#############################################################################


def DiT_Llama_600M_patch2_Spatial(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=1536, n_layers=16, n_heads=32, **kwargs
    )


def DiT_Llama_600M_GQA_patch2(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=1536, n_layers=16, n_heads=32, n_kv_heads=8, **kwargs
    )


def DiT_Llama_2B_patch2(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=2304, n_layers=24, n_heads=32, **kwargs
    )


def DiT_Llama_3B_patch2(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs
    )


def DiT_Llama_7B_patch2(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=4096, n_layers=32, n_heads=32, **kwargs
    )
