"""Own-stock baseline and peer-attention adapter for three stock tasks."""

import math
import typing

import torch
from torch import nn


class OwnStock(nn.Module):
    """Predicts return, EPS-surprise and risk labels from a stock's tokens."""

    def __init__(
        self, features: int, width: int = 32, dropout: float = 0.1
    ) -> None:
        """Initializes the encoder and the three-task head.

        Args:
            features: Width of each stock token.
            width: Embedding width.
            dropout: Dropout rate inside the encoder.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, width),
            nn.GELU(),
        )
        self.head = nn.Linear(width, 3)

    @typing.override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs))


class PeerAdapter(nn.Module):
    """Gated residual on the frozen baseline from attention over graph peers.

    Graph edge weights bias the learned query-key scores rather than replace
    them. Each task has its own query and gate. Inputs are own [B, F], peers
    [B, K, F] and edge weights [B, K], where weight zero marks padding.
    """

    def __init__(
        self,
        features: int,
        width: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initializes attention, gate and residual layers.

        Args:
            features: Width of each stock token.
            width: Embedding width, split across the heads.
            heads: Number of attention heads.
            dropout: Dropout rate inside the residual head.
        """
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.encoder = nn.Sequential(
            nn.Linear(features, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.query = nn.Linear(width, width * 3)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.log_prior_strength = nn.Parameter(torch.zeros(heads))
        self.task_embedding = nn.Parameter(torch.randn(3, width) * 0.02)
        self.delta = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        self.gate = nn.Linear(width * 2, 1)
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        nn.init.constant_(self.gate.bias, -1)

    @typing.override
    def forward(
        self, own: torch.Tensor, peers: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        batch, count, _ = peers.shape
        valid = weights > 0
        # Mask raw tokens before encoding so padding cannot reach the output.
        peers = torch.where(valid[..., None], peers, torch.zeros_like(peers))
        own_embedding = self.encoder(own)
        peer_embedding = self.encoder(peers)
        dimension = self.width // self.heads
        heads = (self.heads, dimension)
        query = self.query(own_embedding).view(batch, 3, *heads)
        keys = self.key(peer_embedding).view(batch, count, *heads)
        values = self.value(peer_embedding - own_embedding[:, None]).view(
            batch, count, *heads
        )
        logits = torch.einsum("bthd,bkhd->bthk", query, keys)
        logits = logits / math.sqrt(dimension)
        strength = self.log_prior_strength.exp().clamp(max=10)
        prior = weights.clamp_min(1e-8).log()[:, None, None, :]
        logits = logits + prior * strength[None, None, :, None]
        logits = logits.masked_fill(~valid[:, None, None, :], -1e9)
        attention = logits.softmax(-1) * valid[:, None, None, :]
        pooled = torch.einsum("bthk,bkhd->bthd", attention, values)
        pooled = pooled.reshape(batch, 3, self.width)
        task_own = own_embedding[:, None] + self.task_embedding[None]
        joint = torch.cat([task_own, pooled], dim=-1)
        gate = self.gate(joint).sigmoid().squeeze(-1)
        delta = self.delta(joint).squeeze(-1)
        return delta * gate * valid.any(-1, keepdim=True)
