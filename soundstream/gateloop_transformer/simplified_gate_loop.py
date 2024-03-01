from functools import partial
from math import prod
from typing import Tuple

from .associative_scan import associative_scan
from .gateloop_transformer import RMSNorm

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.nn import Cell

# plain pytorch non-fused associative scan


def exists(v):
    return v is not None


def abs_clamp_eps(t, eps=1e-20):
    sign = ops.sign(t)
    return sign * t.abs().clamp(min=eps)


# associative scan using heinsen sequences
# https://github.com/glassroom/heinsen_sequence
# graciously shared to the world by Franz A. Heinsen in https://arxiv.org/abs/2311.06281 in October 2023


def heinsen_associative_scan(a, kv, eps=1e-20):
    log_a = a.clamp(min=eps).log()
    log_kv = abs_clamp_eps(kv, eps=eps).to(ms.complex64).log()

    a_star = log_a.cumsum(axis=1)
    log_x0_plus_b_star = (log_kv - a_star).logcumsumexp(aixs=1)
    log_x = a_star + log_x0_plus_b_star
    return a_star.exp().real(), log_x.exp().real()


# naive associative scan with some torchscript of binary operator


def binary_operator(a: Tuple[Tensor, Tensor], b: Tuple[Tensor, Tensor]):
    a_i, kv_i = a
    a_j, kv_j = b
    return a_j * a_i, ops.addcmul(kv_j, a_j, kv_i)


# gate loop operator


def gate_loop_operator(q, kv, a, cache=None, heinsen=False):
    if exists(cache):
        cache_a, cache_kv = cache
        a = ops.cat([cache_a, a], 1)
        kv = ops.cat([cache_kv, kv], 1)

    if heinsen:
        a, kv = heinsen_associative_scan(a, kv)
    else:
        a = a.to(ms.float32)
        kv = kv.to(ms.float32)
        a, kv = associative_scan(binary_operator, (a, kv))

    if exists(cache):
        a = a.split(2, 1)
        kv = kv.split(2, 1)

    return q * kv, (a[:, -1], kv[:, -1])


# simple gate loop layer


class SimpleGateLoopLayer(Cell):
    """
    simplified gate loop
    seeing if it can supplement attention as shown in https://github.com/lucidrains/mega-pytorch
    """

    def __init__(self, dim, prenorm=True, use_heinsen=False, post_ln=False, reverse=False):
        super().__init__()

        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.dim = dim

        self.to_qkva = nn.Dense(dim, dim * 3, has_bias=False)

        self.use_heinsen = use_heinsen

        if use_heinsen:
            self.gate_loop_fn = partial(gate_loop_operator, heinsen=True)
        else:
            self.gate_loop_fn = gate_loop_operator

        self.maybe_post_ln = nn.LayerNorm((dim,), epsilon=1e-5) if post_ln else nn.Identity()

        self.reverse = reverse

    def construct(self, x, cache=None, return_cache=False):
        if self.reverse:
            x = ops.flip(x, dims=(-2,))

        x = self.norm(x)

        qkva = self.to_qkva(x)
        qkva = qkva.reshape((qkva.shape[:2] + (3, int(qkva.shape[-1] / 3)))).movedim(2, 0).movedim(3, 2).expand_dims(-1)
        q, kv, a = qkva.reshape((3, prod(qkva.shape[1:-2])) + qkva.shape[-2:])

        out, cache = self.gate_loop_fn(q, kv, a.sigmoid(), cache=cache)

        out = out.reshape((int(out.shape[0] / self.dim), self.dim, out.shape[1], 1)).squeeze(-1).movedim(1, 2)
        out = self.maybe_post_ln(out)

        if self.reverse:
            out = ops.flip(out, dims=(-2,))

        if not return_cache:
            return out

        assert not self.reverse, "caching only works with non-reversed seq"

        return out, cache
