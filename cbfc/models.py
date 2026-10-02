"""Models. `units()` lists parameter names per depth unit; splits and LR groups use it."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def identity(x):
    return x


ACTIVATIONS = {"relu": F.relu, "tanh": torch.tanh, "gelu": F.gelu, "linear": identity}


def _init(layer, activation, output=False):
    if output or activation == "linear":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
    elif activation == "tanh":
        gain = nn.init.calculate_gain("tanh")
        nn.init.xavier_normal_(layer.weight, gain=gain)
    else:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _units(model, first, blocks_attr, last):
    """[first unit, one unit per block in model.<blocks_attr>, last unit]."""
    all_names = [name for name, _ in model.named_parameters()]
    n_blocks = len(getattr(model, blocks_attr))
    units = [first]
    for i in range(n_blocks):
        prefix = f"{blocks_attr}.{i}."
        units.append([name for name in all_names if name.startswith(prefix)])
    units.append(last)
    return units


class MLP(nn.Module):
    """`depth` hidden layers; res_scale > 0 adds h + res_scale * act(W h) after layer 0."""

    def __init__(self, in_dim, classes, depth, width=128, activation="relu", layernorm=False,
                 dropout=0.0, res_scale=0.0):
        super().__init__()
        dims = [in_dim] + [width] * depth + [classes]
        self.layers = nn.ModuleList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(d_in, d_out))
        if layernorm:
            self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))
        else:
            self.norms = None
        self.act = ACTIVATIONS[activation]
        self.drop = nn.Dropout(dropout)
        self.res_scale = res_scale
        for i, layer in enumerate(self.layers):
            is_output = (i == depth)
            _init(layer, activation, is_output)

    def layer_map(self, i, h):
        z = self.layers[i](h)
        if self.norms is not None:
            z = self.norms[i](z)
        z = self.drop(self.act(z))
        use_residual = self.res_scale > 0 and i > 0
        if use_residual:
            return h + self.res_scale * z
        return z

    def forward(self, x, layer=None, upper_from=None, return_hidden=False):
        """layer / upper_from / return_hidden expose sub-maps for Jacobian baselines."""
        if layer is not None:
            return self.layer_map(layer, x)
        if upper_from is None:
            h = x.flatten(1)
            first = 0
        else:
            h = x
            first = upper_from
        hidden = []
        for i in range(first, len(self.layers) - 1):
            h = self.layer_map(i, h)
            hidden.append(h)
        out = self.layers[-1](h)
        if return_hidden:
            return out, hidden
        return out

    def units(self):
        units = []
        for i in range(len(self.layers)):
            unit = [f"layers.{i}.weight", f"layers.{i}.bias"]
            has_norm = self.norms is not None and i < len(self.norms)
            if has_norm:
                unit += [f"norms.{i}.weight", f"norms.{i}.bias"]
            units.append(unit)
        return units


def make_norm(norm, width, groups):
    if norm == "groupnorm":
        return nn.GroupNorm(min(groups, width), width)
    if norm == "batchnorm":
        return nn.BatchNorm2d(width)
    if norm == "none":
        return nn.Identity()
    raise KeyError(norm)


class ConvBlock(nn.Module):
    def __init__(self, width, kernel, activation, norm, groups, dropout, skip):
        super().__init__()
        self.conv = nn.Conv2d(width, width, kernel, padding=kernel // 2)
        self.norm = make_norm(norm, width, groups)
        self.act = ACTIVATIONS[activation]
        self.drop = nn.Dropout2d(dropout)
        self.skip = skip

    def forward(self, x):
        y = self.norm(self.conv(x))
        if self.skip:
            y = y + x
        return self.drop(self.act(y))


class ResCNN(nn.Module):
    """Uniform width and resolution: every block is an equivalent depth unit."""

    def __init__(self, in_channels, classes, depth, width=64, kernel=3, activation="relu",
                 norm="groupnorm", groups=8, dropout=0.0, skip=False):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, kernel, padding=kernel // 2)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ConvBlock(width, kernel, activation, norm, groups, dropout, skip))
        self.head = nn.Linear(width, classes)
        conv_init = "relu" if activation in ("relu", "gelu") else "linear"
        _init(self.stem, conv_init)
        for block in self.blocks:
            _init(block.conv, conv_init)
        _init(self.head, activation, True)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean((2, 3))
        return self.head(pooled)

    def units(self):
        return _units(self, ["stem.weight", "stem.bias"], "blocks", ["head.weight", "head.bias"])


SMALLCNN = [(32, 1, False), (32, 2, False), (32, 1, True), (64, 2, False), (64, 1, True)]


class SmallCNN(nn.Module):
    def __init__(self, in_channels=3, classes=10, skip=False):
        super().__init__()
        self.convs = nn.ModuleList()
        channels_in = in_channels
        for channels_out, stride, _ in SMALLCNN:
            self.convs.append(nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1))
            channels_in = channels_out
        self.head = nn.Linear(channels_in, classes)
        self.skip = skip
        for conv in self.convs:
            _init(conv, "relu")

    def forward(self, x):
        for (_, _, has_residual), conv in zip(SMALLCNN, self.convs):
            a = F.relu(conv(x))
            if self.skip and has_residual:
                x = x + a
            else:
                x = a
        pooled = x.mean((2, 3))
        return self.head(pooled)

    def units(self):
        units = [[f"convs.{i}.weight", f"convs.{i}.bias"] for i in range(len(self.convs))]
        units.append(["head.weight", "head.bias"])
        return units


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, causal=False):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads
        self.p = dropout
        self.causal = causal

    def forward(self, x):
        B, T, C = x.shape
        head_dim = C // self.heads
        qkv = self.qkv(x).view(B, T, 3, self.heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dropout_p = self.p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=self.causal)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0, causal=False):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, causal)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        h = F.gelu(self.fc1(self.n2(x)))
        h = self.fc2(self.drop(h))
        return x + self.drop(h)


class ViT(nn.Module):
    fisher_exclude = ("cls", "pos")

    def __init__(self, in_channels, classes, depth=12, dim=192, heads=3, patch=4, image=32, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        n_patches = (image // patch) ** 2
        self.patch = nn.Conv2d(in_channels, dim, patch, patch)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(len(tokens), -1, -1)
        x = torch.cat([cls, tokens], 1) + self.pos
        for block in self.blocks:
            x = block(x)
        cls_out = self.norm(x)[:, 0]
        return self.head(cls_out)

    def units(self):
        return _units(self, ["patch.weight", "patch.bias", "cls", "pos"], "blocks",
                      ["norm.weight", "norm.bias", "head.weight", "head.bias"])


class CausalLM(nn.Module):
    fisher_exclude = ("tok.weight", "pos")

    def __init__(self, vocab, length, depth=8, dim=512, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, length, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, dropout, causal=True) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        length = x.shape[1]
        z = self.drop(self.tok(x) + self.pos[:, :length])
        for block in self.blocks:
            z = block(z)
        return self.head(self.norm(z))

    def units(self):
        return _units(self, ["tok.weight", "pos"], "blocks", ["norm.weight", "norm.bias", "head.weight"])


def build(arch, shape, classes, **kw):
    channels, height, _ = shape
    if arch == "mlp":
        return MLP(channels * height * height, classes, **kw)
    if arch == "rescnn":
        return ResCNN(channels, classes, **kw)
    if arch == "smallcnn":
        return SmallCNN(channels, classes, **kw)
    if arch == "vit":
        return ViT(channels, classes, image=height, **kw)
    raise KeyError(arch)
