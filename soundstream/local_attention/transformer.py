from math import prod

from .local_attention import LocalAttention

import mindspore as ms
from mindspore import Parameter, nn, ops

# helper function


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def l2norm(t):
    return ops.L2Normalize(-1)(t)


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.set_train(False)
        out = fn(model, *args, **kwargs)
        model.set_train(was_training)
        return out

    return inner


# sampling functions


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = ops.topk(logits, k)
    probs = ops.full_like(logits, float("-inf"))
    probs = probs.scatter(1, ind, val)
    return probs


# multi-head attention


class LocalMHA(nn.Cell):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head=64,
        heads=8,
        dropout=0.0,
        causal=False,
        prenorm=False,
        qk_rmsnorm=False,
        qk_scale=8,
        use_xpos=False,
        xpos_scale_base=None,
        exact_windowsize=None,
        gate_values_per_head=False,
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm((dim,), epsilon=1e-5) if prenorm else None

        self.heads = heads
        self.to_qkv = nn.Dense(dim, inner_dim * 3, has_bias=False)
        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = Parameter(ops.ones(dim_head))
            self.k_scale = Parameter(ops.ones(dim_head))

        self.attn_fn = LocalAttention(
            dim=dim_head,
            window_size=window_size,
            causal=causal,
            autopad=True,
            scale=(qk_scale if qk_rmsnorm else None),
            exact_windowsize=default(exact_windowsize, True),
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            **kwargs
        )

        self.to_v_gate = None

        if gate_values_per_head:
            self.to_v_gate = nn.SequentialCell(nn.Dense(dim, heads))

        self.to_out = nn.Dense(inner_dim, dim, has_bias=False)

    def construct(self, x, mask=None, attn_bias=None):
        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, axis=-1)
        q, k, v = map(
            lambda t: t.reshape(t.shape[:2] + (self.heads, int(t.shape[-1] / self.heads))).movedim(1, 2), (q, k, v)
        )

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        out = self.attn_fn(q, k, v, mask=mask, attn_bias=attn_bias)

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            gates = gates.movedim(1, 2).expand_dims(-1)
            out = out * gates.sigmoid()

        out = out.movedim(1, 2)
        out = out.reshape(out.shape[:2] + (prod(out.shape[-2:]),))
        return self.to_out(out)


# feedforward


class GEGLU(nn.Cell):
    def construct(self, x):
        x, gate = x.chunk(2, axis=-1)
        return x * ops.gelu(gate)


def FeedForward(dim, mult=4, dropout=0.0):
    inner_dim = int(dim * mult * 2 / 3)

    return nn.SequentialCell(
        nn.LayerNorm((dim,), epsilon=1e-5),
        nn.Dense(dim, inner_dim * 2, has_bias=False),
        GEGLU(),
        nn.Dropout(p=dropout),
        nn.Dense(inner_dim, dim, has_bias=False),
    )


# dynamic positional bias


class DynamicPositionBias(nn.Cell):
    def __init__(self, dim, heads):
        super().__init__()
        self.mlp = nn.SequentialCell(nn.Dense(1, dim), nn.SiLU(), nn.Dense(dim, dim), nn.SiLU(), nn.Dense(dim, heads))

    def construct(self, i, j):
        assert j >= i

        rel_dist = ops.arange(j, dtype=ms.float32)
        bias = self.mlp(rel_dist.expand_dims(-1))

        i_seq = ops.arange(j - i, j)
        j_seq = ops.arange(j)

        rel_dist_indices = (i_seq.expand_dims(-1) - j_seq.expand_dims(-2)).abs()

        bias = bias[rel_dist_indices].movedim(-1, 0)
        return bias


# main transformer class


class LocalTransformer(nn.Cell):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        causal=True,
        local_attn_window_size=512,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ignore_index=-1,
        use_xpos=False,
        xpos_scale_base=None,
        use_dynamic_pos_bias=False,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len
        self.layers = nn.CellList([])

        self.local_attn_window_size = local_attn_window_size
        self.dynamic_pos_bias = None
        if use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(dim=dim // 2, heads=heads)

        for _ in range(depth):
            self.layers.append(
                nn.CellList(
                    [
                        LocalMHA(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            causal=causal,
                            window_size=local_attn_window_size,
                            use_xpos=use_xpos,
                            xpos_scale_base=xpos_scale_base,
                            use_rotary_pos_emb=not use_dynamic_pos_bias,
                            prenorm=True,
                            **kwargs
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.ignore_index = ignore_index
        self.to_logits = nn.SequentialCell(
            nn.LayerNorm((dim,), epsilon=1e-5), nn.Dense(dim, num_tokens, has_bias=False)
        )

    @eval_decorator
    def generate(self, prime, seq_len, temperature=1.0, filter_thres=0.9, **kwargs):
        n = prime.shape[1]

        out = prime

        for _ in range(seq_len):
            logits = self.forward(out[:, -self.max_seq_len :], **kwargs)
            filtered_logits = top_k(logits[:, -1], thres=filter_thres)
            probs = ops.softmax(filtered_logits / temperature, axis=-1)
            sampled = ops.multinomial(probs, 1, replacement=False)
            out = ops.cat((out, sampled.to(out.dtype)), axis=-1)

        return out[:, n:]

    def construct(self, x, mask=None, return_loss=False):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        n = x.shape[1]
        x = self.token_emb(x)

        assert n <= self.max_seq_len
        x = x + self.pos_emb(ops.arange(n))

        # dynamic pos bias

        attn_bias = None
        if exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)

        # go through layers

        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_bias=attn_bias) + x
            x = ff(x) + x

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = logits.movedim(-1, -2)
        loss = ops.cross_entropy(logits, labels, ignore_index=self.ignore_index)
        return loss
