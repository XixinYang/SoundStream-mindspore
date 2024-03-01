from math import prod

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.nn import Cell, CellList

from .associative_scan import associative_scan
from .rotary_embedding import RotaryEmbedding
from .utils import einsum_ms

# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def Sequential(*modules):
    modules = list(filter(exists, modules))
    num_modules = len(modules)

    if num_modules == 0:
        return nn.Identity()
    elif num_modules == 1:
        return modules[0]

    return nn.SequentialCell(*modules)


# rms norm


class RMSNorm(Cell):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = Parameter(ops.ones(dim))

    def construct(self, x):
        return ops.L2Normalize(-1)(x) * self.scale * self.gamma


# norm wrappers


class PreNorm(Cell):
    def __init__(self, dim, fn: Cell):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def construct(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class PostNorm(Cell):
    def __init__(self, dim, fn: Cell):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm((dim,), epsilon=1e-5)

    def construct(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)


# feedforward


def FeedForward(dim, mult=4):
    dim_inner = dim * mult
    return nn.SequentialCell(nn.Dense(dim, dim_inner), nn.GELU(approximate=False), nn.Dense(dim_inner, dim))


# attention


class CausalFullAttention(Cell):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        rotary_emb=False,
        add_swish_gating=False,
        data_dependent_rel_pos=False,
        frac_gradient_data_dependent_rel_pos=0.5,
        softmax_normalize=None,
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads
        self.softmax_normalize = default(softmax_normalize, not data_dependent_rel_pos)

        self.scale = dim_head**-0.5

        self.rotary_emb = RotaryEmbedding(dim_head) if rotary_emb else None

        self.to_qkv = nn.Dense(dim, dim_inner * 3, bias=False)

        self.data_dependent_rel_pos = data_dependent_rel_pos
        self.frac_gradient_data_dependent_rel_pos = frac_gradient_data_dependent_rel_pos

        if data_dependent_rel_pos:
            self.to_a = nn.Dense(dim, dim_inner, bias=False)

        self.to_gates = None

        if add_swish_gating:
            self.to_gates = nn.SequentialCell(nn.Dense(dim, dim_inner, bias=False), nn.SiLU())

        self.to_out = nn.Dense(dim_inner, dim)

    def construct(self, x, ablate_complex=False, ablate_state_transition=False):
        qkv = self.to_qkv(x)
        q, k, v = (
            qkv.reshape(qkv.shape[:2] + (3, self.heads, int(qkv.shape[-1] / 3 / self.heads)))
            .movedim(2, 0)
            .movedim(2, 3)
        )

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        q = q * self.scale

        if self.data_dependent_rel_pos and not ablate_state_transition:
            frac_gradient = self.frac_gradient_data_dependent_rel_pos

            a = self.to_a(x)
            a = a.reshape((a.shape[0], a.shape[1], self.heads, int(a.shape[-1] / 2 / self.heads), 2)).movedim(1, 2)

            # allow for data dependent relative position projection to change more slowly
            # alternative to using rotary_embedding lowered learning rate mentioned in paper

            a = a * frac_gradient + Tensor(a) * (1 - frac_gradient)

            a = ops.Complex(a[:, 0], a[:, 1])

            if ablate_complex:
                a = a.real() + 0.0j

            magnitude, phase = ops.ComplexAbs()(a.abs()), a.angle()
            a = ops.polar(magnitude.sigmoid(), phase)

            a = a.expand_dims(-1)
            a_cumprod = a.cumprod(dim=-2)

            a_cumprod_real = a_cumprod.real().clamp(min=1e-10)
            a_cumprod_real_inverse = 1.0 / a_cumprod_real

            q, k = map(lambda t: t.reshape(t.shape[:-1] + (int(t.shape[-1] / 2), 2)), (q, k))

            q = q * a_cumprod_real
            k = k * a_cumprod_real_inverse

            q, k = map(lambda t: t.reshap(t.shape[:-2] + (prod(t.shape[-2:]),)), (q, k))

        sim = einsum_ms("b h i d, b h j d -> b h i j", q, k)

        i, j = sim.shape[2:]
        causal_mask = ops.ones((i, j), dtype=ms.bool_).triu(j - i + 1)

        if self.softmax_normalize:
            sim = sim.masked_fill(causal_mask, -np.finfo(sim.asnumpy().dtype).max)
            attn = ops.softmax(sim, axis=-1)
        else:
            attn = sim.masked_fill(causal_mask, 0.0)

        out = einsum_ms("b h i j, b h j d -> b h i d", attn, v)

        if exists(self.to_gates):
            out = out * self.to_gates(x)
            out = out.reshape((out.shape[0], out.shape[1], self.heads, int(out.shape[-1] / self.heads))).movedim(1, 2)

        out = out.movedim(1, 2)
        out = out.reshape((out.shape[0], out.shape[1], out.shape[2] * out.shape[3]))

        return self.to_out(out)


# data gated linear attention with "gateloop operator"


def gate_loop_operator(q, k, v, a):
    """
    the pseudocode in section 3.2 of the paper
    """

    kv = einsum_ms("b n d, b n e -> b n d e", k, v)
    kv = kv.to(ms.complex64)

    def binary_operator(a, b):
        a_i, kv_i = a
        a_j, kv_j = b
        return a_j * a_i, a_j * kv_i + kv_j

    # a=a.to(ms.float32)
    _, kv = associative_scan(binary_operator, (a, kv))

    return einsum_ms("b n d, b n d e -> b n e", q, kv.real())


class GateLoopedAttention(Cell):
    def __init__(
        self, dim, heads=None, dim_inner=None, add_swish_gating=True, sub_ln=False, frac_gradient_state_transition=0.9
    ):
        super().__init__()
        self.frac_gradient_state_transition = frac_gradient_state_transition

        dim_inner = default(dim_inner, dim)
        heads = default(heads, dim_inner)

        self.heads = heads
        assert (
            dim_inner % heads
        ) == 0, (
            f"dimension for gate looped attention {dim_inner} must be divisible by number of gate loop heads {heads}"
        )

        self.to_qkv = nn.Dense(dim, dim_inner * 3, has_bias=False)

        self.to_a = nn.Dense(dim, heads * 2)

        self.maybe_sub_ln = nn.LayerNorm((dim_inner,), epsilon=1e-5) if sub_ln else nn.Identity()

        self.to_gates = None

        if add_swish_gating:
            self.to_gates = nn.SequentialCell(nn.Dense(dim, dim_inner, has_bias=False), nn.SiLU())

        self.to_out = (
            nn.Dense(dim_inner, dim, has_bias=False) if dim_inner != dim or add_swish_gating else nn.Identity()
        )

    def construct(self, x, ablate_complex=False, ablate_state_transition=False):
        frac_gradient = self.frac_gradient_state_transition

        q, k, v = self.to_qkv(x).chunk(3, axis=-1)

        q, k, v = map(
            lambda t: t.reshape((t.shape[:2] + (self.heads, int(t.shape[-1] / self.heads)))).movedim(1, 2), (q, k, v)
        )
        q, k, v = map(lambda t: t.reshape((prod(t.shape[:2]),) + t.shape[2:]), (q, k, v))

        a = self.to_a(x)
        a = a.reshape(a.shape[:2] + (self.heads, 2)).movedim(1, 2).expand_dims(3).expand_dims(4)
        a = a.reshape((prod(a.shape[:2]),) + a.shape[2:])
        a = a * frac_gradient + Tensor(a) * (1 - frac_gradient)
        a = a.chunk(2, -1)
        a = ops.Complex()(a[0], a[1]).squeeze(-1)

        if ablate_complex:
            a = a.real() + 0.0j

        if ablate_state_transition:
            a = ops.ones_like(a.real()) + 0.0j
        else:
            # activations for state transitions
            # sigmoid for magnitude, identity for phase

            magnitude, phase = ops.ComplexAbs()(a.abs()), a.angle()
            a = ops.polar(magnitude.sigmoid(), phase)

        fn = gate_loop_operator

        out = fn(q, k, v, a)

        out = out.reshape((int(out.shape[0] / self.heads), self.heads) + out.shape[1:]).movedim(1, 2)
        out = out.reshape(out.shape[:2] + (prod(out.shape[2:]),))

        out = self.maybe_sub_ln(out)

        if exists(self.to_gates):
            out = self.to_gates(x) * out

        return self.to_out(out)


# main class


class Transformer(Cell):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_gate_looped_attn=True,
        gate_loop_heads=None,
        attn_add_swish_gating=True,
        dim_gate_looped_attn=None,
        attn_softmax_normalize=None,
        data_dependent_rel_pos=False,
        frac_gradient_state_transition=0.9,
        ablate_complex=False,
        ablate_state_transition=False,
        rotary_emb=False,
        post_ln_norm=False,
        sub_ln=False,
    ):
        super().__init__()
        self.ablate_complex = ablate_complex
        self.ablate_state_transition = ablate_state_transition

        self.token_emb = nn.Embedding(num_tokens, dim)

        layers = CellList([])

        layer_wrapper = PreNorm if not post_ln_norm else PostNorm

        for _ in range(depth):
            if use_gate_looped_attn:
                spatial_mixer = GateLoopedAttention(
                    dim=dim,
                    heads=gate_loop_heads,
                    dim_inner=dim_gate_looped_attn,
                    add_swish_gating=attn_add_swish_gating,
                    sub_ln=sub_ln,
                    frac_gradient_state_transition=frac_gradient_state_transition,
                )
            else:
                spatial_mixer = CausalFullAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    rotary_emb=rotary_emb,
                    add_swish_gating=attn_add_swish_gating,
                    softmax_normalize=attn_softmax_normalize,
                    data_dependent_rel_pos=data_dependent_rel_pos,
                    frac_gradient_data_dependent_rel_pos=frac_gradient_state_transition,
                )

            channelwise_mixer = FeedForward(dim=dim, mult=ff_mult)

            layers.append(CellList([layer_wrapper(dim, spatial_mixer), layer_wrapper(dim, channelwise_mixer)]))

        self.layers = layers

        self.to_logits = Sequential(
            RMSNorm(dim) if not post_ln_norm else None, nn.Dense(dim, num_tokens, has_bias=False)
        )

    def construct(self, x, return_loss=False, ablate_complex=None, ablate_state_transition=None):
        ablate_complex = default(ablate_complex, self.ablate_complex)
        ablate_state_transition = default(ablate_state_transition, self.ablate_state_transition)

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(x, ablate_complex=ablate_complex, ablate_state_transition=ablate_state_transition)

            x = ff(x)

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = logits.movedim(-1, 1)
        return ops.cross_entropy(logits, labels)
