"""Explicit Transformer reference for candidate-set context encoding.

Input E: [batch_of_queries, candidates, embedding_dim]. Each batch row contains
only candidates from ONE query. No candidate positional encoding is added.
This block is executable PyTorch; the full tree encoder/trainer is still a draft.
"""
import math
import torch
from torch import nn


class CandidateTransformer(nn.Module):
    def __init__(self, embedding_dim=64, heads=4, ff_dim=None, dropout=0.1):
        super().__init__()
        if heads <= 0 or embedding_dim <= 0 or embedding_dim % heads:
            raise ValueError("embedding_dim must be positive and divisible by heads")
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim // heads
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim or 4 * embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim or 4 * embedding_dim, embedding_dim),
        )

    def split_heads(self, tensor):
        # [B, N, D] -> [B, N, heads, head_dim] -> [B, heads, N, head_dim]
        batch, count, _ = tensor.shape
        return tensor.reshape(batch, count, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, E, valid_mask=None):
        if E.ndim != 3 or E.shape[-1] != self.embedding_dim or E.shape[1] == 0:
            raise ValueError("Expected nonempty candidate embeddings [B, N, D]")
        if valid_mask is None:
            valid_mask = torch.ones(E.shape[:2], dtype=torch.bool, device=E.device)
        if valid_mask.shape != E.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [B, N]; True means real candidate")
        if valid_mask.device != E.device:
            raise ValueError("Mask and embeddings must be on the same device")
        if not valid_mask.any(dim=1).all():
            raise ValueError("Each query must have at least one real candidate")

        # Clear padded inputs before projection; padding must not affect outputs.
        E = E.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        # Step 1: Learn distinct query, key, and value projections.
        Q = self.split_heads(self.query(E))
        K = self.split_heads(self.key(E))
        V = self.split_heads(self.value(E))

        # Step 2: Pairwise candidate compatibility, independently for each query.
        # scores[b, h, i, j] = dot(Q[b,h,i], K[b,h,j]) / sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 3: Exclude padded keys before normalization along candidate j.
        key_mask = valid_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        # Step 4: Weighted candidate context for each head, followed by fusion.
        context = torch.matmul(weights, V)  # [B, heads, N, head_dim]
        context = context.transpose(1, 2).contiguous().reshape(E.shape)
        A = self.output(context)

        # Step 5: Preserve each candidate's structural embedding through residuals.
        H = self.norm1(E + self.residual_dropout(A))

        # Step 6: Shared positionwise feed-forward network and second residual.
        H = self.norm2(H + self.residual_dropout(self.feed_forward(H)))

        # Padded query rows are not predictions and must not enter the loss.
        return H.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
