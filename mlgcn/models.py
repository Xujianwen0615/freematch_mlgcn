# -*- coding: utf-8 -*-
"""
ML-GCN model definitions adapted for the private multi-label dataset.

Source / reference:
  - ML-GCN (megvii-research/ML-GCN): https://github.com/megvii-research/ML-GCN.git
  - The prior (already working) implementation in D:\\freematch_mlgcn\\semilearn\\nets\\models_mlgcn.py

Provided components
-------------------
  * GCNResnet   — the graph-based classifier (ML-GCN) applied on top of a ResNet backbone.
  * build_model — factory that builds any of: 'gcn' (ML-GCN), 'gap' (plain ResNet+GMP) or 'gmp' (plain ResNet+GMP).
                  'gap' / 'gmp' serve as ablation baselines (no label-correlation graph + no GCN head).

Key design notes
----------------
  * Backbone is configurable (resnet18/34/50/101). The GCN output dim is tied to the backbone
    feature dim (2048 for resnet50/101, 512 for resnet18/34) so that the diagram dimension matches.
  * Adjacency matrix is pre-computed from ``adj_file`` (a pickle ``{"P": [C, C]}`` conditional-probability
    matrix), binarised with threshold ``t``, re-weighted by the ML-GCN paper rule and sym-normed,
    then registered as a non-trainable buffer.
  * ``inp`` (optional GloVe word embeddings [C, in_channel]) is used as the *learnable* initial label
    embedding fed into the GCN.
  * GMP (global max pooling) is used by default, matching the original ML-GCN paper.
"""
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter

# ---------------------------------------------------------------------------
# Adjacency matrix construction (copied / lightly tuned from models_mlgcn.py)
# ---------------------------------------------------------------------------
def gen_A(num_classes, t, adj_file):
    """Read ``{"P": [C, C]}``, binarise at threshold ``t`` and re-weight per ML-GCN."""
    with open(adj_file, "rb") as f:
        result = pickle.load(f)
    P = result["P"]                                   # conditional probability matrix, values in [0, 1]
    A = (P >= t).astype(np.float32)
    nums = np.sum(A, axis=1, dtype=np.float32).reshape(-1, 1)
    _adj = A
    _nums = nums

    # ML-GCN paper re-weighting: neighbour weight = p / degree, self loop = 1 - p
    p = 0.2
    for i in range(num_classes):
        if _nums[i] > 0:
            _adj[i] = _adj[i] * (p / _nums[i])
        _adj[i, i] = 1.0 - p
    return _adj


def gen_adj(A):
    """Standard symmetric GCN normalisation: D^{-1/2} A D^{-1/2}."""
    D = torch.sum(A, dim=1)
    D = torch.pow(D, -0.5)
    D[torch.isinf(D)] = 0.0
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(D, A), D)
    return adj


class GraphConvolution(nn.Module):
    """GCN layer with optional (broadcastable) bias, as used by ML-GCN."""

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class GCNResnet(nn.Module):
    """ML-GCN: ResNet backbone -> GMP -> matmul(label graph features)."""

    def __init__(self, model, num_classes, in_channel=300, t=0.4,
                 adj_file=None, inp=None, gcn_hidden=1024, feature_dim=2048,
                 pooling="max", mean=None, std=None):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )
        self.num_classes = num_classes
        # ML-GCN paper uses global max pooling (GMP).
        self.pooling = nn.AdaptiveMaxPool2d(1) if pooling == "max" else nn.AdaptiveAvgPool2d(1)

        self.gc1 = GraphConvolution(in_channel, gcn_hidden)
        self.gc2 = GraphConvolution(gcn_hidden, feature_dim)
        self.relu = nn.LeakyReLU(0.2)

        # Pre-compute + register the normalised adjacency matrix (non-trainable buffer).
        if adj_file is not None:
            _adj = gen_A(num_classes, t, adj_file)
            A = torch.from_numpy(_adj).float()
            self.register_buffer("A", A)
            adj_final = gen_adj(A).detach()
            self.register_buffer("adj", adj_final)
        else:
            # Fallback: identity adjacency (no label-correlation graph used).
            A = torch.eye(num_classes)
            self.register_buffer("A", A)
            self.register_buffer("adj", torch.eye(num_classes))

        # Word embeddings as learnable parameters (GloVe or random).
        if inp is not None:
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp).float()
            self.inp = nn.Parameter(inp.clone(), requires_grad=True)
        else:
            self.inp = nn.Parameter(torch.randn(num_classes, in_channel), requires_grad=True)
            print("[models] WARNING: random label embedding used; pass GloVe for best results.")

        self.image_normalization_mean = mean if mean is not None else [0.2318, 0.2274, 0.2761]
        self.image_normalization_std = std if std is not None else [0.2077, 0.2220, 0.2262]

    def forward(self, x):
        feature = self.features(x)                        # (B, feature_dim, h, w)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)       # (B, feature_dim)

        inp = self.inp                                    # (C, in_channel)
        adj = self.adj                                    # (C, C)

        x_gcn = self.gc1(inp, adj)                        # (C, gcn_hidden)
        x_gcn = self.relu(x_gcn)
        x_gcn = self.gc2(x_gcn, adj)                      # (C, feature_dim)
        x_gcn = x_gcn.transpose(0, 1)                     # (feature_dim, C)

        logits = torch.matmul(feature, x_gcn)             # (B, C)
        return logits

    def get_config_optim(self, lr, lrp):
        return [
            {"params": self.features.parameters(), "lr": lr * lrp},
            {"params": self.gc1.parameters(), "lr": lr},
            {"params": self.gc2.parameters(), "lr": lr},
            {"params": self.inp, "lr": lr},
        ]


# ---------------------------------------------------------------------------
# Plain baselines (for ablations): same backbone + GMP/GAP, linear classifier.
# ---------------------------------------------------------------------------
class PlainResNet(nn.Module):
    """Linear-head baseline: ResNet backbone -> GMP -> Linear, with optional bias flag."""

    def __init__(self, model, num_classes, pooling="max", bias=True,
                 mean=None, std=None):
        super(PlainResNet, self).__init__()
        self.features = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.AdaptiveMaxPool2d(1) if pooling == "max" else nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self._feat_dim(model), num_classes, bias=bias)
        self.reset_parameters(layer=self.fc)
        self.image_normalization_mean = mean if mean is not None else [0.2318, 0.2274, 0.2761]
        self.image_normalization_std = std if std is not None else [0.2077, 0.2220, 0.2262]

    @staticmethod
    def _feat_dim(model):
        # Read the out-channels of the last block (layer4).
        try:
            return model.layer4[-1].conv3.out_channels
        except Exception:
            return model.fc.in_features

    @staticmethod
    def reset_parameters(layer):
        stdv = 1.0 / math.sqrt(layer.weight.size(1))
        layer.weight.data.uniform_(-stdv, stdv)
        if layer.bias is not None:
            layer.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        feature = self.features(x)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        return self.fc(feature)

    def get_config_optim(self, lr, lrp):
        return [
            {"params": self.features.parameters(), "lr": lr * lrp},
            {"params": self.fc.parameters(), "lr": lr},
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
BACKBONE_ARCHS = ("resnet18", "resnet34", "resnet50", "resnet101")


def _backbone(name, pretrained=True):
    if name not in BACKBONE_ARCHS:
        raise ValueError(f"Unsupported backbone '{name}'. Choose one of {BACKBONE_ARCHS}.")
    fn = getattr(models, name)
    # Use the modern torchvision `weights` API to avoid the `pretrained` deprecation.
    try:
        if pretrained:
            return fn(weights=f"IMAGENET1K_V1")
        return fn(weights=None)
    except TypeError:
        # Older torchvision fallback.
        return fn(pretrained=pretrained)


def build_model(num_classes, backbone="resnet50", variant="gcn",
                pretrained=True, t=0.4, adj_file=None, inp=None,
                in_channel=300, gcn_hidden=1024, feature_dim=2048,
                pooling="max", mean=None, std=None):
    """
    Build a model by ``variant``:
      * 'gcn' → ML-GCN (graph head, default)
      * 'gap' → plain ResNet + global average pooling + Linear  (ablation baseline)
      * 'gmp' → plain ResNet + global max pooling + Linear      (ablation baseline)
    """
    prefix = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}
    feat_dim = prefix[backbone] if backbone in prefix else feature_dim

    model = _backbone(backbone, pretrained=pretrained)

    if variant == "gcn":
        return GCNResnet(model, num_classes,
                         in_channel=in_channel, t=t, adj_file=adj_file, inp=inp,
                         gcn_hidden=gcn_hidden, feature_dim=feat_dim,
                         pooling=pooling, mean=mean, std=std)
    elif variant in ("gap", "gmp"):
        return PlainResNet(model, num_classes,
                           pooling=("avg" if variant == "gap" else "max"),
                           bias=(variant == "gap"), mean=mean, std=std)
    else:
        raise ValueError(f"Unknown variant '{variant}'.")


def load_class_embeddings(inp_file):
    """Load the GloVe-style class embeddings [C, d] from a pickle file."""
    with open(inp_file, "rb") as f:
        inp = pickle.load(f)
    return np.asarray(inp, dtype=np.float32)


if __name__ == "__main__":
    # Quick smoke test.
    import torch
    adj_file = r"D:\CV\custom_muti\custom_adj.pkl"
    inp_file = r"D:\CV\custom_muti\custom_glove_word2vec.pkl"
    inp = load_class_embeddings(inp_file)
    m = build_model(num_classes=15, backbone="resnet50", variant="gcn",
                    pretrained=False, t=0.4, adj_file=adj_file, inp=inp)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print("GCN output:", out.shape, "| params:", sum(p.numel() for p in m.parameters()))
    for v in ("gap", "gmp"):
        m2 = build_model(num_classes=15, backbone="resnet50", variant=v, pretrained=False)
        print(f"{v} output:", m2(x).shape)
