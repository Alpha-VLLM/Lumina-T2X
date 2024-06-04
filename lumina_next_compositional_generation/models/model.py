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
import math
from typing import List, Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, ParallelEmbedding, RowParallelLinear
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .components import RMSNorm


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


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
                frequency_embedding_size,
                hidden_size,
                bias=True,
                gather_output=False,
                init_method=functools.partial(nn.init.normal_, std=0.02),
            ),
            nn.SiLU(),
            RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=True,
                input_is_parallel=True,
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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
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
            num_classes + use_cfg_embedding,
            hidden_size,
            init_method=functools.partial(nn.init.normal_, std=0.02),
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
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
#                               Core NextDiT Model                              #
#############################################################################


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_norm: bool,
        y_dim: int,
    ):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = ColumnParallelLinear(
            dim,
            n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )
        self.wk = ColumnParallelLinear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )
        self.wv = ColumnParallelLinear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )
        if y_dim > 0:
            self.wk_y = ColumnParallelLinear(
                y_dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=nn.init.xavier_uniform_,
            )
            self.wv_y = ColumnParallelLinear(
                y_dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=nn.init.xavier_uniform_,
            )
            self.gate = nn.Parameter(torch.zeros([self.n_local_heads]))

        self.wo = RowParallelLinear(
            n_heads * self.head_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=nn.init.xavier_uniform_,
        )

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_local_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_local_kv_heads * self.head_dim)
            if y_dim > 0:
                self.ky_norm = nn.LayerNorm(self.n_local_kv_heads * self.head_dim)
            else:
                self.ky_norm = nn.Identity()
        else:
            self.q_norm = self.k_norm = nn.Identity()
            self.ky_norm = nn.Identity()

        # for proportional attention computation
        self.base_seqlen = None
        self.proportional_attn = False

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
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(
        x_in: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensors using the given frequency
        tensor.

        This function applies rotary embeddings to the given query 'xq' and
        key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
        input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors
        contain rotary embeddings and are returned as real tensors.

        Args:
            x_in (torch.Tensor): Query or Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
                exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
                and key tensor with rotary embeddings.
        """
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    # copied from huggingface modeling_llama.py
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return (
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
            )

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_local_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Args:
            x:
            x_mask:
            freqs_cis:
            y:
            y_mask:

        Returns:

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = Attention.apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = Attention.apply_rotary_emb(xk, freqs_cis=freqs_cis)

        xq, xk = xq.to(dtype), xk.to(dtype)

        if self.proportional_attn:
            softmax_scale = math.sqrt(math.log(seqlen, self.base_seqlen) / self.head_dim)
        else:
            softmax_scale = math.sqrt(1 / self.head_dim)

        if dtype in [torch.float16, torch.bfloat16]:
            # begin var_len flash attn
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(xq, xk, xv, x_mask, seqlen)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0.0,
                causal=False,
                softmax_scale=softmax_scale,
            )
            output = pad_input(attn_output_unpad, indices_q, bsz, seqlen)
            # end var_len_flash_attn

        else:
            output = (
                F.scaled_dot_product_attention(
                    xq.permute(0, 2, 1, 3),
                    xk.permute(0, 2, 1, 3),
                    xv.permute(0, 2, 1, 3),
                    attn_mask=x_mask.bool().view(bsz, 1, 1, seqlen).expand(-1, self.n_local_heads, seqlen, -1),
                    scale=softmax_scale,
                )
                .permute(0, 2, 1, 3)
                .to(dtype)
            )

        if hasattr(self, "wk_y"):
            num_y = y.shape[0]
            xq = torch.cat([xq[0].unsqueeze(0).repeat(num_y - 1, 1, 1, 1), xq[-1].unsqueeze(0)], dim=0)
            yk = self.ky_norm(self.wk_y(y)).view(num_y, -1, self.n_local_kv_heads, self.head_dim)
            yv = self.wv_y(y).view(num_y, -1, self.n_local_kv_heads, self.head_dim)
            y_mask_in = y_mask.view(num_y, 1, 1, -1).repeat(1, self.n_local_heads, seqlen, 1)
            if region_mask is not None:
                region_mask_in = region_mask.view(num_y, 1, seqlen, 1).repeat(
                    1, self.n_local_heads, 1, y_mask_in.shape[-1]
                )
                y_mask_in = y_mask_in & region_mask_in
            n_rep = self.n_local_heads // self.n_local_kv_heads
            if n_rep >= 1:
                yk = yk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                yv = yv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output_y = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                yk.permute(0, 2, 1, 3),
                yv.permute(0, 2, 1, 3),
                y_mask_in,
            ).permute(0, 2, 1, 3)
            output_y = torch.nan_to_num(output_y)
            output_y = output_y * self.gate.tanh().view(1, 1, -1, 1)
            output_y_cond = torch.sum(output_y[:-1], dim=0, keepdim=True)
            output_y_uncond = torch.sum(output_y[-1:], dim=0, keepdim=True)
            output_y = torch.cat([output_y_cond, output_y_uncond], dim=0)
            output = output + output_y

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
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=nn.init.xavier_uniform_,
        )
        self.w3 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=False,
            init_method=nn.init.xavier_uniform_,
        )

    # @torch.compile
    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        y_dim: int,
    ) -> None:
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
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                min(dim, 1024),
                4 * dim,
                bias=True,
                gather_output=True,
                init_method=nn.init.zeros_,
            ),
        )

        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
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
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa),
                    x_mask,
                    freqs_cis,
                    self.attention_y_norm(y),
                    y_mask,
                    region_mask,
                )
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp),
                )
            )

        else:
            x = x + self.attention_norm2(
                self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    freqs_cis,
                    self.attention_y_norm(y),
                    y_mask,
                    region_mask,
                )
            )
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class ParallelFinalLayer(nn.Module):
    """
    The final layer of NextDiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = ColumnParallelLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            init_method=nn.init.zeros_,
            gather_output=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                min(hidden_size, 1024),
                hidden_size,
                bias=True,
                init_method=nn.init.zeros_,
                gather_output=True,
            ),
        )

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        # shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x


class NextDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        learn_sigma: bool = True,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size

        self.x_embedder = ColumnParallelLinear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
            gather_output=True,
            init_method=nn.init.xavier_uniform_,
        )
        nn.init.constant_(self.x_embedder.bias, 0.0)

        self.t_embedder = ParallelTimestepEmbedder(min(dim, 1024))
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(cap_feat_dim),
            ColumnParallelLinear(
                cap_feat_dim,
                min(dim, 1024),
                bias=True,
                gather_output=True,
                init_method=nn.init.zeros_,
            ),
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    cap_feat_dim,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = ParallelFinalLayer(dim, patch_size, self.out_channels)

        assert (dim // n_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"
        self.freqs_cis = NextDiT.precompute_freqs_cis(
            dim // n_heads,
            384,
            scale_factor=scale_factor,
        )
        self.dim = dim
        self.n_heads = n_heads
        self.scale_factor = scale_factor
        self.pad_token = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.pad_token, std=0.02)

    def unpatchify(self, x: torch.Tensor, img_size: List[Tuple[int, int]], return_tensor=False) -> List[torch.Tensor]:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        pH = pW = self.patch_size
        if return_tensor:
            H, W = img_size[0]
            B = x.size(0)
            L = (H // pH) * (W // pW)
            x = x[:, :L].view(B, H // pH, W // pW, pH, pW, self.out_channels)
            x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
            return x
        else:
            imgs = []
            for i in range(x.size(0)):
                H, W = img_size[i]
                L = (H // pH) * (W // pW)
                imgs.append(
                    x[i][:L]
                    .view(H // pH, W // pW, pH, pW, self.out_channels)
                    .permute(4, 0, 2, 1, 3)
                    .flatten(3, 4)
                    .flatten(1, 2)
                )
        return imgs

    def patchify_and_embed(
        self, x: List[torch.Tensor] | torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
        self.freqs_cis = self.freqs_cis.to(x[0].device)
        if isinstance(x, torch.Tensor):
            pH = pW = self.patch_size
            B, C, H, W = x.size()
            x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3)
            x = self.x_embedder(x)
            x = x.flatten(1, 2)

            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.int32, device=x.device)

            return (
                x,
                mask,
                [(H, W)] * B,
                self.freqs_cis[: H // pH, : W // pW].flatten(0, 1).unsqueeze(0),
            )
        else:
            pH = pW = self.patch_size
            x_embed = []
            freqs_cis = []
            img_size = []
            l_effective_seq_len = []

            for img in x:
                C, H, W = img.size()
                item_freqs_cis = self.freqs_cis[: H // pH, : W // pW]
                freqs_cis.append(item_freqs_cis.flatten(0, 1))
                img_size.append((H, W))
                img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 0, 2, 4).flatten(2)
                img = self.x_embedder(img)
                img = img.flatten(0, 1)
                l_effective_seq_len.append(len(img))
                x_embed.append(img)

            max_seq_len = max(l_effective_seq_len)
            mask = torch.zeros(len(x), max_seq_len, dtype=torch.int32, device=x[0].device)
            padded_x_embed = []
            padded_freqs_cis = []
            for i, (item_embed, item_freqs_cis, item_seq_len) in enumerate(
                zip(x_embed, freqs_cis, l_effective_seq_len)
            ):
                item_embed = torch.cat(
                    [
                        item_embed,
                        self.pad_token.view(1, -1).expand(max_seq_len - item_seq_len, -1),
                    ],
                    dim=0,
                )
                item_freqs_cis = torch.cat(
                    [
                        item_freqs_cis,
                        item_freqs_cis[-1:].expand(max_seq_len - item_seq_len, -1),
                    ],
                    dim=0,
                )
                padded_x_embed.append(item_embed)
                padded_freqs_cis.append(item_freqs_cis)
                mask[i][:item_seq_len] = 1

            x_embed = torch.stack(padded_x_embed, dim=0)
            freqs_cis = torch.stack(padded_freqs_cis, dim=0)
            return x_embed, mask, img_size, freqs_cis

    def forward(
        self, x, t, cap_feats, cap_mask, global_cap_feats=None, global_cap_mask=None, h_split_num=1, w_split_num=1
    ):
        """
        Forward pass of NextDiT.
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        B, C, H, W = x.size()
        x_is_tensor = isinstance(x, torch.Tensor)
        x, mask, img_size, freqs_cis = self.patchify_and_embed(x)
        freqs_cis = freqs_cis.to(x.device)

        t = self.t_embedder(t)  # (N, D)
        cap_mask_float = global_cap_mask.float().unsqueeze(-1)
        cap_feats_pool = (global_cap_feats * cap_mask_float).sum(dim=1) / cap_mask_float.sum(dim=1)
        cap_feats_pool = cap_feats_pool.to(cap_feats)
        cap_emb = self.cap_embedder(cap_feats_pool)
        adaln_input = t + cap_emb

        region_mask = torch.zeros(
            cap_feats.shape[0], H // self.patch_size, W // self.patch_size, dtype=torch.float, device=x.device
        )
        h_patch_size, w_patch_size = H // h_split_num // self.patch_size, W // w_split_num // self.patch_size
        for h_split in range(h_split_num):
            for w_split in range(w_split_num):
                region_id = (h_split + 1) * (w_split + 1) - 1
                region_mask[
                    region_id,
                    h_patch_size * h_split : h_patch_size * (h_split + 1),
                    w_patch_size * w_split : w_patch_size * (w_split + 1),
                ] = 1
        region_mask[-1, :, :] = 1
        region_mask = region_mask.flatten(1, 2)
        region_mask = region_mask > 0.5

        cap_mask = cap_mask.bool()
        for layer in self.layers:
            x = layer(x, mask, freqs_cis, cap_feats, cap_mask, adaln_input=adaln_input, region_mask=region_mask)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, return_tensor=x_is_tensor)
        if self.learn_sigma:
            if x_is_tensor:
                x, _ = x.chunk(2, dim=1)
            else:
                x = [_.chunk(2, dim=0)[0] for _ in x]
        return x

    def forward_with_cfg(
        self,
        x,
        t,
        cap_feats,
        cap_mask,
        cfg_scale,
        scale_factor=1.0,
        scale_watershed=1.0,
        base_seqlen: Optional[int] = None,
        proportional_attn: bool = False,
        global_cap_feats=None,
        global_cap_mask=None,
        h_split_num=1,
        w_split_num=1,
    ):
        """
        Forward pass of NextDiT, but also batches the unconditional forward pass
        for classifier-free guidance.
        """
        # # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        self.freqs_cis = NextDiT.precompute_freqs_cis(
            self.dim // self.n_heads,
            384,
            scale_factor=scale_factor,
            scale_watershed=scale_watershed,
            timestep=t[0].item(),
        )

        if proportional_attn:
            assert base_seqlen is not None
            for layer in self.layers:
                layer.attention.base_seqlen = base_seqlen
                layer.attention.proportional_attn = proportional_attn
        else:
            for layer in self.layers:
                layer.attention.base_seqlen = None
                layer.attention.proportional_attn = proportional_attn

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self(combined, t, cap_feats, cap_mask, global_cap_feats, global_cap_mask, h_split_num, w_split_num)
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
        scale_factor: float = 1.0,
        scale_watershed: float = 1.0,
        timestep: float = 1.0,
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

        if timestep < scale_watershed:
            linear_factor = scale_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scale_factor

        theta = theta * ntk_factor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float().cuda() / dim)) / linear_factor

        timestep = torch.arange(end, device=freqs.device, dtype=torch.float)  # type: ignore

        freqs = torch.outer(timestep, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

        freqs_cis_h = freqs_cis.view(end, 1, dim // 4, 1).repeat(1, end, 1, 1)
        freqs_cis_w = freqs_cis.view(1, end, dim // 4, 1).repeat(end, 1, 1, 1)
        freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1).flatten(2)

        return freqs_cis

    def parameter_count(self) -> int:
        tensor_parallel_module_list = (
            ColumnParallelLinear,
            RowParallelLinear,
            ParallelEmbedding,
        )
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            is_tp_module = isinstance(module, tensor_parallel_module_list)
            for param in module.parameters(recurse=False):
                total_params += param.numel() * (fs_init.get_model_parallel_world_size() if is_tp_module else 1)
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


#############################################################################
#                                 NextDiT Configs                               #
#############################################################################
def NextDiT_2B_patch2(**kwargs):
    return NextDiT(patch_size=2, dim=2304, n_layers=24, n_heads=32, **kwargs)


def NextDiT_2B_GQA_patch2(**kwargs):
    return NextDiT(patch_size=2, dim=2304, n_layers=24, n_heads=32, n_kv_heads=8, **kwargs)
