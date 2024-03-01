from math import pi, prod

from beartype import beartype
from beartype.typing import Literal, Optional, Union

import mindspore as ms
from mindspore import Parameter, Tensor, ops
from mindspore.nn import Cell

from .utils import einsum_ms

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# broadcat, as tortoise-tts was using it


def broadcat(tensors, dim=-1):
    broadcasted_tensors = ms.numpy.broadcast_arrays(*tensors)
    return ops.cat(broadcasted_tensors, axis=dim)


# rotary embedding helper functions


def rotate_half(x):
    x = x.reshape(x.shape[:-1] + (int(x.shape[-1] / 2), 2))
    x1, x2 = x.unbind(dim=-1)
    x = ops.stack((-x2, x1), axis=-1)
    return x.reshape(x.shape[:-2] + (prod(x.shape[-2:]),))


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    t = t.to(ms.float32)
    freqs = freqs.to(ms.float32)
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return ops.cat((t_left, t, t_right), axis=-1)


# learned rotation helpers


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum_ms("..., f -> ... f", rotations, freq_ranges)
        rotations = rotations.reshape(rotations.shape[:-2] + (prod(rotations.shape[-2:]),))

    rotations = rotations.repeat(2, axis=-1)
    return apply_rotary_emb(rotations, t, start_index=start_index)


# classes


class RotaryEmbedding(Cell):
    @beartype
    def __init__(
        self,
        dim,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Union[Literal["lang"], Literal["pixel"], Literal["constant"]] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (ops.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = ops.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = ops.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self.cached_freqs = None
        self.cached_scales = None

        self.freqs = Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.dummy = Tensor(0)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos
        if not use_xpos:
            self.scale = None
            return

        scale = (ops.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.scale = scale

    def get_seq_pos(self, seq_len, dtype, offset=0):
        return (ops.arange(seq_len, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, freq_seq_len=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert (
            not self.use_xpos
        ), "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"

        dtype, seq_len = t.dtype, t.shape[seq_dim]

        if exists(freq_seq_len):
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len

        freqs = self.construct(self.get_seq_pos(seq_len, dtype=dtype, offset=offset), seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = freqs.expand_dims(1)

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len
        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, freq_seq_len=k_len)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        dtype, seq_len = q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype)

        freqs = self.construct(seq, seq_len=seq_len)
        scale = self.get_scale(seq.to(ms.float32), seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = freqs.expand_dims(1)
            scale = scale.expand_dims(1)

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    @beartype
    def get_scale(self, t: Tensor, seq_len: Optional[int] = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and exists(seq_len)

        if should_cache and exists(self.cached_scales) and (seq_len + offset) <= self.cached_scales.shape[0]:
            return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** power.expand_dims(-1)
            scale = ops.cat((scale, scale), axis=-1)

        if should_cache:
            self.cached_scales = scale

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = ops.linspace(-1, 1, steps=dim)
            else:
                pos = ops.arange(dim)

            freqs = self.construct(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = ms.numpy.broadcast_arrays(*all_freqs)
        return ops.cat(all_freqs, axis=-1)

    def construct(self, t: Tensor, seq_len=None, offset=0):
        t = t.to(ms.float32)
        should_cache = (
            self.cache_if_possible and not self.learned_freq and exists(seq_len) and self.freqs_for != "pixel"
        )

        if should_cache and exists(self.cached_freqs) and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return Tensor(self.cached_freqs[offset : (offset + seq_len)])

        freqs = self.freqs.to(ms.float32)

        freqs = einsum_ms("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = freqs.repeat(2, -1)

        if should_cache:
            self.cached_freqs = Tensor(freqs)

        return freqs
