from mindspore import nn, ops

from .utils import einsum_ms


def exists(val):
    return val is not None


class SinusoidalEmbeddings(nn.Cell):
    def __init__(self, dim, scale_base=None, use_xpos=False):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (ops.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq

        # xpos related

        self.use_xpos = use_xpos
        self.scale_base = scale_base

        assert not (use_xpos and not exists(scale_base)), "scale base must be defined if using xpos"

        scale = (ops.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale = scale

    def construct(self, x):
        seq_len = x.shape[-2]

        t = ops.arange(seq_len).to(self.inv_freq.dtype)
        freqs = einsum_ms("i , j -> i j", t, self.inv_freq)
        freqs = ops.cat((freqs, freqs), axis=-1)

        if not self.use_xpos:
            return freqs, ops.ones(1)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** power.expand_dims(-1)
        scale = ops.cat((scale, scale), axis=-1)

        return freqs, scale


def rotate_half(x):
    x = x.reshape(x.shape[:-1] + (2, int(x.shape[-1] / 2)))
    x1, x2 = x.unbind(dim=-2)
    return ops.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, freqs, scale=1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]

    inv_scale = scale**-1

    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k
