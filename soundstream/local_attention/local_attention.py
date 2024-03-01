import math

import numpy as np
from .rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

import mindspore as ms
from mindspore import nn, ops

from .utils import einsum_ms

# constant

TOKEN_SELF_ATTN_VALUE = -5e4

# helper functions


def exists(val):
    return val is not None


def default(value, d):
    return d if not exists(value) else value


def max_neg_value(tensor):
    return -np.finfo(tensor.asnumpy().dtype).max


def l2norm(tensor):
    dtype = tensor.dtype
    normed = ops.L2Normalize(-1)(tensor)
    return normed.type(dtype)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, ops.pad(tensor, (*pad_offset, 0, remainder), value=value)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = ops.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [padded_x[:, ind : (ind + t), ...] for ind in range(forward + backward + 1)]
    return ops.cat(tensors, axis=dim)


# main class


class LocalAttention(nn.Cell):
    def __init__(
        self,
        window_size,
        causal=False,
        look_backward=1,
        look_forward=None,
        dropout=0.0,
        shared_qk=False,
        rel_pos_emb_config=None,
        dim=None,
        autopad=False,
        exact_windowsize=False,
        scale=None,
        use_rotary_pos_emb=True,
        use_xpos=False,
        xpos_scale_base=None,
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), "you cannot look forward if causal"

        self.scale = scale

        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = nn.Dropout(p=dropout)

        self.shared_qk = shared_qk

        # relative positions

        self.rel_pos = None
        self.use_xpos = use_xpos

        if use_rotary_pos_emb and (
            exists(rel_pos_emb_config) or exists(dim)
        ):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]

            self.rel_pos = SinusoidalEmbeddings(
                dim, use_xpos=use_xpos, scale_base=default(xpos_scale_base, window_size // 2)
            )

    def construct(self, q, k, v, mask=None, input_mask=None, attn_bias=None, window_size=None):
        mask = default(mask, input_mask)

        assert not (
            exists(window_size) and not self.use_xpos
        ), "cannot perform window size extrapolation if xpos is not turned on"

        shape, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = (
            q.shape,
            self.autopad,
            -1,
            default(window_size, self.window_size),
            self.causal,
            self.look_backward,
            self.look_forward,
            self.shared_qk,
        )

        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        q, k, v = map(lambda t: t.reshape((math.prod(t.shape[:-2]), t.shape[-2], t.shape[-1])), (q, k, v))

        # auto padding

        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim=-2), (q, k, v))

        b, n, dim_head, dtype = *q.shape, q.dtype

        scale = default(self.scale, dim_head**-0.5)

        assert (
            n % window_size
        ) == 0, f"sequence length {n} must be divisible by window size {window_size} for local attention"

        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = ops.arange(n)
        b_t = seq.reshape((1, windows, window_size))
        # bucketing

        bq, bk, bv = map(lambda t: t.reshape((t.shape[0], windows, int(t.shape[1] / windows), t.shape[-1])), (q, k, v))

        bq = bq * scale

        look_around_kwargs = dict(backward=look_backward, forward=look_forward, pad_value=pad_value)

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        # rotary embeddings

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale=xpos_scale)

        # calculate positions for masking

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = bq_t.expand_dims(-1)
        bq_k = bq_k.expand_dims(-2)

        pad_mask = bq_k == pad_value

        sim = einsum_ms("b h i e, b h j e -> b h i j", bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = attn_bias.expand_dims(1)
            attn_bias = attn_bias.repeat(b // heads, 0)
            sim = sim + attn_bias

        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = self.window_size * self.look_backward
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value

        if not causal and self.exact_windowsize:
            max_backward_window_size = self.window_size * self.look_backward
            max_forward_window_size = self.window_size * self.look_forward
            window_mask = (
                ((bq_k - max_forward_window_size) > bq_t) | (bq_t > (bq_k + max_backward_window_size)) | pad_mask
            )
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]

            if autopad:
                _, mask = pad_to_multiple(mask, window_size, dim=-1, value=False)

            mask_shape = mask.shape
            mask = mask.reshape((int(math.prod(mask_shape) / mask_shape[-1]), windows, window_size))
            mask = look_around(mask, **{**look_around_kwargs, "pad_value": False})
            mask = mask.expand_dims(-2)
            mask = mask.to(ms.int8).repeat(h, 0).to(ms.bool_)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention

        attn = ops.softmax(sim, axis=-1)
        attn = self.dropout(attn)

        # aggregation

        out = einsum_ms("b h i j, b h j e -> b h i e", attn, bv)
        out_shape = out.shape
        out = out.reshape((out_shape[0], int(math.prod(out_shape[1:][:-1])), out_shape[-1]))

        if autopad:
            out = out[:, :orig_seq_len, :]

        out = out.reshape(shape[:-2] + out.shape[-2:])
        return out
