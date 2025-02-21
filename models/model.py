import logging
import math
from typing import Callable, List, Optional, Tuple

from einops import rearrange
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
import torch.nn as nn
import torch.nn.functional as F

from utils.parallel import (
    all_to_all,
    gather_from_sequence_parallel_region,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    scatter_to_sequence_parallel_region,
)
from .components import RMSNorm

logger = logging.getLogger(__name__)


#############################################################################
#             Embedding Layers for Timesteps and Class Labels               #
#############################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
                bias=True,
            ),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

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
        with torch.amp.autocast("cuda", enabled=False):
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


#############################################################################
#                     Core MultiScaleNextDiT Modules                        #
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
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wq.weight)
        self.wk = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wk.weight)
        self.wv = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wv.weight)

        assert y_dim > 0
        self.wk_y = nn.Linear(
            y_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wk_y.weight)
        self.wv_y = nn.Linear(
            y_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wv_y.weight)
        self.gate = nn.Parameter(torch.zeros([self.n_heads]))

        self.wo = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wo.weight)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
            if y_dim > 0:
                self.ky_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
            else:
                self.ky_norm = nn.Identity()
        else:
            self.q_norm = self.k_norm = nn.Identity()
            self.ky_norm = nn.Identity()

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
        with torch.amp.autocast("cuda", enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(1)
            x_out = torch.view_as_real(x * freqs_cis).flatten(2)
            return x_out.type_as(x_in)

    def forward(
        self,
        x_shard: torch.Tensor,
        pad_num: int,
        x_freqs_cis_shard: torch.Tensor,
        y: torch.Tensor,
        x_cu_seqlens: torch.Tensor,
        y_cu_seqlens: torch.Tensor,
        x_max_item_seqlen: int,
        y_max_item_seqlen: int,
    ) -> torch.Tensor:

        seqlen_shard, _ = x_shard.shape
        xq_shard, xk_shard, xv_shard = self.wq(x_shard), self.wk(x_shard), self.wv(x_shard)
        dtype = xq_shard.dtype

        xq_shard = self.q_norm(xq_shard)
        xk_shard = self.k_norm(xk_shard)

        xq_shard = xq_shard.view(seqlen_shard, self.n_heads, self.head_dim)
        xk_shard = xk_shard.view(seqlen_shard, self.n_kv_heads, self.head_dim)
        xv_shard = xv_shard.view(seqlen_shard, self.n_kv_heads, self.head_dim)

        xq_shard = Attention.apply_rotary_emb(xq_shard, freqs_cis=x_freqs_cis_shard)  # note use sharded rope
        xk_shard = Attention.apply_rotary_emb(xk_shard, freqs_cis=x_freqs_cis_shard)

        xq_shard, xk_shard = xq_shard.to(dtype), xk_shard.to(dtype)

        sp_group = get_sequence_parallel_group()

        xq_gather = all_to_all(xq_shard, gather_dim=0, gather_pad=pad_num, scatter_dim=1, group=sp_group)
        xk_gather = all_to_all(xk_shard, gather_dim=0, gather_pad=pad_num, scatter_dim=1, group=sp_group)
        xv_gather = all_to_all(xv_shard, gather_dim=0, gather_pad=pad_num, scatter_dim=1, group=sp_group)

        softmax_scale = math.sqrt(1 / self.head_dim)

        assert dtype in [torch.float16, torch.bfloat16]
        assert y.dtype == dtype

        output = flash_attn_varlen_func(
            xq_gather,
            xk_gather,
            xv_gather,
            cu_seqlens_q=x_cu_seqlens,
            cu_seqlens_k=x_cu_seqlens,
            max_seqlen_q=x_max_item_seqlen,
            max_seqlen_k=x_max_item_seqlen,
            dropout_p=0.0,
            causal=False,
            softmax_scale=softmax_scale,
        )

        sp_world_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        yk = (
            self.ky_norm(self.wk_y(y))
            .view(-1, self.n_kv_heads, self.head_dim)
            .to(dtype)
            .chunk(sp_world_size, dim=1)[sp_rank]
        )
        yv = self.wv_y(y).view(-1, self.n_kv_heads, self.head_dim).chunk(sp_world_size, dim=1)[sp_rank]

        assert yk.dtype == dtype

        output_y = flash_attn_varlen_func(
            xq_gather,
            yk,
            yv,
            cu_seqlens_q=x_cu_seqlens,
            cu_seqlens_k=y_cu_seqlens,
            max_seqlen_q=x_max_item_seqlen,
            max_seqlen_k=y_max_item_seqlen,
            dropout_p=0.0,
            causal=False,
            softmax_scale=softmax_scale,
        )

        output_y = output_y * self.gate.chunk(sp_world_size)[sp_rank].tanh().view(1, -1, 1)

        output = output + output_y

        output = all_to_all(output, gather_dim=1, scatter_dim=0, scatter_pad=pad_num, group=sp_group)

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
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w1.weight)
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w2.weight)
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w3.weight)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    # @torch.compile
    def forward(self, x):
        y = self._forward_silu_gating(self.w1(x), self.w3(x))
        # from flash_attn.ops.activations import swiglu
        # y = swiglu(self.w1(x), self.w3(x))
        return self.w2(y)


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
            nn.Linear(
                min(dim, 1024),
                4 * dim,
                bias=True,
            ),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(
        self,
        x_shard: torch.Tensor,
        pad_num,
        x_src_ids_shard: torch.Tensor,
        x_freqs_cis_shard: torch.Tensor,
        y: torch.Tensor,
        x_cu_seqlens: torch.Tensor,
        y_cu_seqlens: torch.Tensor,
        x_max_item_seqlen: int,
        y_max_item_seqlen: int,
        adaln_input: torch.Tensor,
    ):

        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        x_shard = x_shard + gate_msa[x_src_ids_shard] * self.attention_norm2(
            self.attention(
                self.attention_norm1(x_shard) * scale_msa[x_src_ids_shard],
                pad_num,
                x_freqs_cis_shard,
                self.attention_y_norm(y),
                x_cu_seqlens,
                y_cu_seqlens,
                x_max_item_seqlen,
                y_max_item_seqlen,
            )
        )

        x_shard = x_shard + gate_mlp[x_src_ids_shard] * self.ffn_norm2(
            self.feed_forward(
                self.ffn_norm1(x_shard) * scale_mlp[x_src_ids_shard],
            )
        )

        return x_shard


class FinalLayer(nn.Module):
    """
    The final layer of MultiScaleNextDiT.
    """

    def __init__(self, hidden_size, patch_size, f_patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size,
            f_patch_size * patch_size * patch_size * out_channels,
            bias=True,
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(hidden_size, 1024),
                hidden_size,
                bias=True,
            ),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x_shard, x_src_ids_shard, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x_shard = self.norm_final(x_shard) * scale[x_src_ids_shard]
        x_shard = self.linear(x_shard)
        return x_shard


class MultiScaleNextDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        all_patch_size: Tuple[int] = (2,),
        all_f_patch_size: Tuple[int] = (2,),
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        rope_theta: float = 10000.0,
        t_scale: float = 1.0,
        motion_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.motion_scale = motion_scale

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(
                in_features=f_patch_size * patch_size * patch_size * in_channels,
                out_features=dim,
                bias=True,
            )
            nn.init.xavier_uniform_(x_embedder.weight)
            nn.init.constant_(x_embedder.bias, 0.0)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size, f_patch_size, self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.motion_embedder = TimestepEmbedder(min(dim, 1024))
        nn.init.zeros_(self.motion_embedder.mlp[2].weight)
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(cap_feat_dim),
            nn.Linear(
                cap_feat_dim,
                min(dim, 1024),
                bias=True,
            ),
        )
        nn.init.zeros_(self.cap_embedder[1].weight)
        nn.init.zeros_(self.cap_embedder[1].bias)

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

        head_dim = dim // n_heads

        assert head_dim % 16 == 0

        self.freqs_cis = MultiScaleNextDiT.precompute_freqs_cis(
            head_dim,
            64,
            head_dim // 8 * 2,
            128,
            head_dim // 8 * 3,
            128,
            head_dim // 8 * 3,
            theta=self.rope_theta,
        )
        self.pad_token = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.pad_token, std=0.02)

    def unpatchify(
        self,
        x: torch.Tensor,
        x_item_seqlens: List[int],
        size: List[Tuple],
        patch_size,
        f_patch_size,
        return_tensor=False,
    ) -> List[torch.Tensor] | torch.Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        pH = pW = patch_size
        pF = f_patch_size
        B = len(x_item_seqlens)

        if return_tensor:
            F, H, W = size[0]
            if F == 1:
                F_compute = pF
            else:
                F_compute = F
            L = sum(x_item_seqlens, 0)
            x = x[:, :L].view(B, F_compute // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
            x = rearrange(x, "b f h w pf ph pw c -> b c (f pf) (h ph) (w pw)")[:, :, :F]
            return x
        else:
            L = sum(x_item_seqlens, 0)
            x = torch.split(x[:L], x_item_seqlens, 0)
            imgs = []
            assert len(size) == B
            for i in range(B):
                F, H, W = size[i]
                if F == 1:
                    F_compute = pF
                else:
                    F_compute = F
                assert x_item_seqlens[i] == (F_compute // pF) * (H // pH) * (W // pW)
                imgs.append(
                    rearrange(
                        x[i].view(F_compute // pF, H // pH, W // pW, pF, pH, pW, self.out_channels),
                        "f h w pf ph pw c -> c (f pf) (h ph) (w pw)",
                    )[:, :F]
                )
        return imgs

    def patchify_and_embed(
        self, x: List[torch.Tensor] | torch.Tensor, patch_size, f_patch_size
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[Tuple]]:
        self.freqs_cis = self.freqs_cis.to(x[0].device)

        pH = pW = patch_size
        pF = f_patch_size
        all_video = []
        all_freqs_cis = []
        all_size = []
        all_effective_seq_len = []

        for i, video in enumerate(x):
            C, F, H, W = video.size()
            all_size.append((F, H, W))
            if F == 1 and pF > 1:
                video = video.repeat(1, pF, 1, 1)
                C, F, H, W = video.size()
            item_freqs_cis = self.freqs_cis[: F // pF, : H // pH, : W // pW]
            all_freqs_cis.append(item_freqs_cis.flatten(0, 2))

            video = video.view(C, F // pF, pF, H // pH, pH, W // pW, pW)
            video = rearrange(video, "c f pf h ph w pw -> (f h w) (pf ph pw c)")
            all_video.append(video)

            all_effective_seq_len.append(len(video))

        all_video = torch.cat(all_video, dim=0)

        all_freqs_cis = torch.cat(all_freqs_cis, dim=0)

        return all_video, all_effective_seq_len, all_freqs_cis, all_size

    def forward(self, x, t, cap_feats, cap_mask, motion_score, patch_size, f_patch_size):

        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        x_is_tensor = isinstance(x, torch.Tensor)

        x, x_item_seqlens, x_freqs_cis, size = self.patchify_and_embed(x, patch_size, f_patch_size)
        x_max_item_seqlen = max(x_item_seqlens)
        x_cu_seqlens = F.pad(
            torch.cumsum(
                torch.tensor(x_item_seqlens, dtype=torch.int32, device=x.device), dim=0, dtype=torch.torch.int32
            ),
            (1, 0),
        )
        x_src_ids = torch.cat(
            [torch.full((count,), i, dtype=torch.long, device=x.device) for i, count in enumerate(x_item_seqlens)]
        )

        assert not x.requires_grad
        x_shard, pad_num = scatter_to_sequence_parallel_region(x, rank0_only=False)
        x_shard = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_shard)
        x_src_ids_shard, _ = scatter_to_sequence_parallel_region(x_src_ids, rank0_only=False)
        x_freqs_cis_shard, _ = scatter_to_sequence_parallel_region(x_freqs_cis, rank0_only=False)
        del x, x_freqs_cis

        t = t * self.t_scale
        t = self.t_embedder(t)  # (N, D)

        motion_score = motion_score * self.motion_scale
        motion_mask = motion_score < 0
        motion_score[motion_mask] = 0
        motion = self.motion_embedder(motion_score)
        motion[motion_mask] = 0

        cap_mask_float = cap_mask.to(cap_feats).unsqueeze(-1)
        cap_feats_pool = (cap_feats * cap_mask_float).sum(dim=1) / cap_mask_float.sum(dim=1)
        cap_feats_pool = cap_feats_pool.to(cap_feats)
        cap_emb = self.cap_embedder(cap_feats_pool)

        adaln_input = t + cap_emb + motion

        cap_mask = cap_mask.bool()

        cap_feats, _, cap_cu_seqlens, cap_max_item_seqlen = unpad_input(cap_feats, cap_mask)[:4]

        del cap_mask

        for layer in self.layers:
            x_shard = layer(
                x_shard,
                pad_num,
                x_src_ids_shard,
                x_freqs_cis_shard,
                cap_feats,
                x_cu_seqlens,
                cap_cu_seqlens,
                x_max_item_seqlen,
                cap_max_item_seqlen,
                adaln_input=adaln_input,
            )

        x_shard = self.all_final_layer[f"{patch_size}-{f_patch_size}"](x_shard, x_src_ids_shard, adaln_input)

        x = gather_from_sequence_parallel_region(x_shard, rank0_only=False, pad_num=pad_num)

        x = self.unpatchify(x, x_item_seqlens, size, patch_size, f_patch_size, return_tensor=x_is_tensor)

        return x

    @staticmethod
    def value_from_time_aware_config(config, t):
        if isinstance(config, (float, int, str)):
            return config
        elif isinstance(config, (tuple, list)):
            assert isinstance(config[0], (float, int, str))
            assert all([isinstance(x, (tuple, list, str)) for x in config[1:]])
            result = config[0]
            for thresh, val in config[1:]:
                if t >= thresh:
                    result = val
                else:
                    break
            return result
        else:
            raise ValueError(f"invalid time-aware config {config}")

    def forward_with_multi_cfg(
        self,
        x,
        t,
        cap_feats,
        cap_mask,
        motion_score,
        patch_comb=None,
        cfg_scale=None,
        renorm_cfg=False,
        print_info=False,
    ):

        t_item = t[0].item()

        patch_comb_to_use = self.value_from_time_aware_config(patch_comb, t_item)
        f_patch_size, patch_size = patch_comb_to_use.split("x")
        f_patch_size, patch_size = int(f_patch_size), int(patch_size)

        assert cfg_scale is not None

        assert len(x) % (1 + len(cfg_scale)) == 0
        pos_batch_size = len(x) // (1 + len(cfg_scale))

        cfg_scale = [self.value_from_time_aware_config(config, t_item) for config in cfg_scale]
        motion_score = [self.value_from_time_aware_config(config, t_item) for config in motion_score]
        motion_score = torch.tensor(motion_score).to(x)

        l_cfg_to_use = []
        l_samples = [x[:pos_batch_size]]
        l_t = [t[:pos_batch_size]]
        l_cap_feats = [cap_feats[:pos_batch_size]]
        l_cap_mask = [cap_mask[:pos_batch_size]]
        l_motion_score = [motion_score[:pos_batch_size]]
        for i, single_cfg_scale in enumerate(cfg_scale):
            if single_cfg_scale > 0.0:
                l_cfg_to_use.append(single_cfg_scale)
                neg_start = pos_batch_size * (i + 1)
                neg_end = pos_batch_size * (i + 2)
                l_samples.append(l_samples[0])  # copy the positive sample
                l_t.append(t[neg_start:neg_end])
                l_cap_feats.append(cap_feats[neg_start:neg_end])
                l_cap_mask.append(cap_mask[neg_start:neg_end])
                l_motion_score.append(motion_score[neg_start:neg_end])

        effective_x = torch.cat(l_samples, dim=0)
        t = torch.cat(l_t, dim=0)
        cap_feats = torch.cat(l_cap_feats, dim=0)
        cap_mask = torch.cat(l_cap_mask, dim=0)
        motion_score = torch.cat(l_motion_score, dim=0)
        del l_samples, l_t, l_cap_feats, l_cap_mask, l_motion_score, cfg_scale

        if print_info:
            print(t, effective_x.shape, l_cfg_to_use, motion_score, f"{f_patch_size}x{patch_size}", flush=True)

        model_out = self.forward(effective_x, t, cap_feats, cap_mask, motion_score, patch_size, f_patch_size)

        with torch.amp.autocast("cuda", enabled=False):
            model_out = model_out.float()
            model_out = torch.split(model_out, pos_batch_size, dim=0)
            ori_pos_out, l_neg_out = model_out[0], model_out[1:]

            pos_out = ori_pos_out.clone()
            assert len(l_cfg_to_use) == len(l_neg_out)
            for single_cfg_scale, single_neg_out in zip(l_cfg_to_use, l_neg_out):
                pos_out = pos_out + single_cfg_scale * (ori_pos_out - single_neg_out)

            if float(renorm_cfg) > 0.0:
                ori_pos_norm = torch.linalg.vector_norm(
                    ori_pos_out, dim=tuple(range(1, len(ori_pos_out.shape))), keepdim=True
                )

                max_new_norm = ori_pos_norm * renorm_cfg

                new_pos_norm = torch.linalg.vector_norm(pos_out, dim=tuple(range(1, len(pos_out.shape))), keepdim=True)

                for i in range(pos_batch_size):
                    if new_pos_norm[i] < max_new_norm[i]:
                        if print_info:
                            print(
                                f"item {i} ori_norm {ori_pos_norm[i].squeeze()} "
                                f"post-CFG norm {new_pos_norm[i].squeeze()}, no renorm",
                                flush=True,
                            )
                    else:
                        pos_out[i] = pos_out[i] * (max_new_norm[i] / new_pos_norm[i])
                        if print_info:
                            print(
                                f"item {i} ori_norm {ori_pos_norm[i].squeeze()} "
                                f"post-CFG norm {new_pos_norm[i].squeeze()}, renorm to {max_new_norm[i]}",
                                flush=True,
                            )

            x = x.clone()
            x[:pos_batch_size] = pos_out
            return x

    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        f_max: int,
        f_dim: int,
        h_max: int,
        h_dim: int,
        w_max: int,
        w_dim: int,
        theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ):

        assert f_dim + h_dim + w_dim == dim
        assert f_dim % 2 == 0
        assert h_dim % 2 == 0
        assert w_dim % 2 == 0

        theta = theta * ntk_factor

        l_sub_freqs_cis = []
        for d_max, d_dim in [(f_max, f_dim), (h_max, h_dim), (w_max, w_dim)]:
            freqs = 1.0 / (theta ** (torch.arange(0, d_dim, 2, dtype=torch.float64, device="cpu") / d_dim))
            t = torch.arange(d_max, device=freqs.device, dtype=torch.float64)
            t = t / rope_scaling_factor
            freqs = torch.outer(t, freqs)  # type: ignore
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64

            l_sub_freqs_cis.append(freqs_cis)

        l_sub_freqs_cis[0] = l_sub_freqs_cis[0].view(f_max, 1, 1, f_dim // 2).repeat(1, h_max, w_max, 1)
        l_sub_freqs_cis[1] = l_sub_freqs_cis[1].view(1, h_max, 1, h_dim // 2).repeat(f_max, 1, w_max, 1)
        l_sub_freqs_cis[2] = l_sub_freqs_cis[2].view(1, 1, w_max, w_dim // 2).repeat(f_max, h_max, 1, 1)

        freqs_cis = torch.cat(l_sub_freqs_cis, dim=-1)

        return freqs_cis

    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    def my_fsdp(self, wrap_func: Callable):
        for i, layer in enumerate(self.layers):
            self.layers[i] = wrap_func(layer)

    def my_checkpointing(self):
        for i, layer in enumerate(self.layers):
            self.layers[i] = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    def my_compile(self):
        for i, layer in enumerate(self.layers):
            self.layers[i] = torch.compile(layer)


#############################################################################
#                           Lumina-Video Configs                            #
#############################################################################
def MultiScaleNextDiT_2B(**kwargs):
    return MultiScaleNextDiT(dim=2304, n_layers=24, n_heads=16, **kwargs)


def MultiScaleNextDiT_2B_GQA(**kwargs):
    return MultiScaleNextDiT(dim=2304, n_layers=24, n_heads=16, n_kv_heads=8, **kwargs)


def MultiScaleNextDiT_4B_GQA(**kwargs):
    return MultiScaleNextDiT(dim=3072, n_layers=32, n_heads=24, n_kv_heads=8, **kwargs)
