from abc import ABC

import torch
from torch.nn import Module, Embedding, Linear, ReLU, ModuleList, Sequential, GELU, LayerNorm
from functools import partial
import math


# helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def softmax_kernel(data,
                   projection_matrix,
                   is_query,
                   normalize=True,
                   eps=1e-4,
                   device=None):
    """Non-Negative Softmax Kernel Features
    """

    normalizer = 1. / (data.shape[-1] ** 0.25) if normalize else 1.
    ratio = 1. / (projection_matrix.shape[0] ** 0.5)
    data_mod_shape = data.shape[:(len(data.shape)) - 2] + projection_matrix.shape
    data_thick_random_matrix = torch.zeros(data_mod_shape, device=device) + projection_matrix

    data_dash = torch.einsum('...id,...jd->...ij',
                             (normalizer * data),
                             data_thick_random_matrix)

    diag_data = data ** 2                     # dot product ?
    diag_data = torch.sum(diag_data, dim=-1)  # dot product ?
    diag_data = (diag_data / 2.0) * (normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    _max = torch.max(data_dash,
                     dim=-1,
                     keepdim=True).values if is_query else torch.max(data_dash)
    data_dash = ratio * (torch.exp(data_dash - diag_data - _max) + eps)
    return data_dash


def generalized_kernel(data,
                       projection_matrix,
                       kernel_fn=nn.ReLU(),
                       kernel_epsilon=0.001,
                       normalize=True,
                       device=None):
    data_normalizer = 1.0 / (data.shape[-1] ** 0.25) if normalize else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    data_mod_shape = data.shape[0: len(data.shape) - 2] + projection_matrix.shape
    data_thick_random_matrix = torch.zeros(data_mod_shape, device=device) + projection_matrix

    data_dash = torch.einsum('...id,...jd->...ij',
                             (data_normalizer * data),
                             data_thick_random_matrix)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, _ = torch.qr(unstructured_block.cpu(), some=True)
    q = q.to(device)
    return q.t()


def gaussian_orthogonal_random_matrix(rows,
                                      cols,
                                      scaling=0,
                                      device=None):
    num_full_blocks = int(rows / cols)

    block_list = []

    for _ in range(num_full_blocks):
        q = orthogonal_matrix_chunk(cols, device=device)
        block_list.append(q)

    remaining_rows = rows - num_full_blocks * cols
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(cols, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((rows, cols), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(cols))) * torch.ones((rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd->...ne', context, q)
    return out


# efficient causal linear attention, created by EPFL
def causal_linear_attention(q, k, v):
    return CausalDotProduct.apply(q, k, v)


# inefficient causal linear attention, without cuda code,
# for reader's reference not being used
def causal_linear_attention_noncuda(q, k, v):
    k_cumsum = k.cumsum(dim=-2)
    context = torch.einsum('...nd,...ne->...nde', k, v)
    context = context.cumsum(dim=-3)
    context /= k_cumsum.unsqueeze(dim=-1)
    out = torch.einsum('...nde,...nd->...ne', context, q)
    return out


class FastAttention(Module, ABC):
    def __init__(self,
                 dim_heads,
                 num_features=None,
                 redraw_projection=True,
                 ort_scaling=0,
                 causal=False,
                 generalized_attention=False,
                 kernel_fn=ReLU()):
        super(Module, self).__init__()
        num_features = default(num_features,
                               int(dim_heads * math.log(dim_heads)))

        self.causal = causal
        self.dim_heads = dim_heads
        self.num_features = num_features
        self.ort_scaling = ort_scaling
        self.redraw_projection = redraw_projection

        self.create_projection = partial(gaussian_orthogonal_random_matrix,
                                         rows=self.num_features,
                                         cols=dim_heads,
                                         scaling=ort_scaling)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        if not redraw_projection:
            projection_matrix = self.create_projection()
            self.register_buffer('projection_matrix', projection_matrix)

    def forward(self, q, k, v):
        device = q.device

        if self.redraw_projection:
            projection_matrix = self.create_projection(device=device)
        else:
            projection_matrix = self.projection_matrix

        if self.generalized_attention:
            create_kernel = partial(generalized_kernel,
                                    kernel_fn=self.kernel_fn,
                                    projection_matrix=projection_matrix,
                                    device=device)
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else causal_linear_attention
        out = attn_fn(q, k, v)
        return out


class PreNorm(Module, ABC):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Chunk(Module, ABC):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


class FeedForward(Module, ABC):
    def __init__(self, dim, multiple=4):
        super().__init__()
        self.net = Sequential(
            Linear(dim, dim * multiple),
            GELU(),
            Linear(dim * multiple, dim)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(Module, ABC):
    def __init__(self,
                 dim,
                 causal=False,
                 heads=8,
                 num_features=None,
                 redraw_projection=True,
                 generalized_attention=False,
                 kernel_fn=ReLU()):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.fast_attention = FastAttention(dim // heads,
                                            num_features,
                                            redraw_projection,
                                            causal=causal,
                                            generalized_attention=generalized_attention,
                                            kernel_fn=kernel_fn)

        self.heads = heads
        self.to_qkv = Linear(dim, dim * 3, bias=False)
        self.to_out = Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if exists(mask):
            mask = mask[:, None, :, None]
            k.masked_fill_(~mask, 0)

        out = self.fast_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Performer(Module, ABC):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 causal=False,
                 ff_mult=4,
                 nb_features=None,
                 reversible=False,
                 ff_chunks=1,
                 generalized_attention=False, kernel_fn=ReLU()):
        super().__init__()
        layers = ModuleList([])
        for _ in range(depth):
            layers.append(ModuleList([
                PreNorm(dim, SelfAttention(dim, causal=causal, heads=heads, num_features=nb_features,
                                           generalized_attention=generalized_attention, kernel_fn=kernel_fn)),
                PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, multiple=ff_mult), along_dim=1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}
        self.net = execute_type(layers, args_route={**attn_route_map})

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)


class PerformerLM(Module, ABC):
    def __init__(self,
                 *,
                 num_tokens,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 causal=False,
                 ff_mult=4,
                 nb_features=None,
                 reversible=False,
                 ff_chunks=1,
                 generalized_attention=False,
                 kernel_fn=ReLU()):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = Embedding(num_tokens, dim)
        self.pos_emb = Embedding(max_seq_len, dim)
        self.performer = Performer(dim, depth, heads, causal, ff_mult, nb_features, reversible, ff_chunks,
                                   generalized_attention, kernel_fn)
        self.to_logits = Linear(dim, num_tokens)

    def forward(self, x, **kwargs):
        b, n, device = *x.shape, x.device
        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(n, device=device))
        x = self.performer(x, **kwargs)
        return self.to_logits(x)
