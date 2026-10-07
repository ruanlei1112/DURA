"""DURA candidate-set encoder: implementation-oriented pseudocode.

This file replaces the earlier query-graph encoder. It is deliberately NOT a
trainable Lightning model. Every abstract operation below raises until backed
by tensor code; old training/prediction paths must not silently keep running.
Input is a complete candidate set for ONE query, not a minibatch of unrelated
plans. The explicit attention block is in lcm/candidate_transformer.py.
"""
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class PlanNode:
    operator: str
    relation_ids: Sequence[int]  # schema-wide stable IDs; multiple tables allowed
    is_scan: bool
    has_filter: bool
    scan_selectivity: float | None  # estimate from statistics; never runtime labels


@dataclass(frozen=True)
class CandidatePlan:
    query_id: str
    plan_id: str
    nodes: Sequence[PlanNode]
    children: Any  # tree topology; explicit null children and node mask


def tensor_operation(name, *args, **kwargs):
    """Abstract tensor operation, intentionally not a fake implementation."""
    raise NotImplementedError(f"Pseudocode operation needs implementation: {name}")


def operator_category(operator):
    """Seven physical-operator categories described in PLARQ."""
    categories = {
        "Hash Join": 0, "Merge Join": 1, "Nested Loop": 2,
        "Seq Scan": 3, "Parallel Seq Scan": 3,
        "Bitmap Heap Scan": 4, "Bitmap Index Scan": 4,
        "Index Scan": 5, "Index Only Scan": 5,
    }
    return categories.get(operator, 6)


def node_features(node, relation_count):
    """x_v = concat(operator_one_hot, relation_multi_hot, scan_selectivity)."""
    operator = [0.0] * 7
    operator[operator_category(node.operator)] = 1.0
    relations = [0.0] * relation_count
    for relation in node.relation_ids:
        if not 0 <= relation < relation_count:
            raise ValueError("Relation ID is outside the fixed schema vocabulary")
        relations[relation] = 1.0
    selectivity = 0.0
    if node.is_scan and node.has_filter:
        if node.scan_selectivity is None or not 0 <= node.scan_selectivity <= 1:
            raise ValueError("Filtered scans need estimated selectivity in [0, 1]")
        selectivity = float(node.scan_selectivity)
    return operator + relations + [selectivity]


class CandidateSetEncoder:
    """Pseudocode: per-tree convolution followed by cross-plan Transformer."""

    def __init__(self, relation_count, embedding_dim=64, attention_heads=4, context_encoder=None):
        if relation_count <= 0 or attention_heads <= 0 or embedding_dim <= 0:
            raise ValueError("Dimensions and head count must be positive")
        if embedding_dim % attention_heads:
            raise ValueError("Embedding dimension must divide evenly into heads")
        self.relation_count = relation_count
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.context_encoder = context_encoder

    def encode_tree(self, plan):
        if not plan.nodes:
            raise ValueError("A candidate plan must contain a tree")
        X = [node_features(node, self.relation_count) for node in plan.nodes]
        # Shared trainable weights across plans; three layers aggregate each
        # parent/left-child/right-child subtree. Missing children use zeros.
        for layer in range(3):
            X = tensor_operation("TreeConv+LayerNorm+ReLU", X, plan.children,
                                 shared_layer=layer, output_dim=self.embedding_dim)
        # Pool REAL tree nodes only: padding must not influence max/mean pooling.
        return tensor_operation("MaskedDynamicPooling", X, plan.children)

    def forward(self, candidates):
        if not candidates:
            raise ValueError("A query must have at least one candidate")
        if len({p.query_id for p in candidates}) != 1:
            raise ValueError("Cross-query attention is forbidden")
        if len({p.plan_id for p in candidates}) != len(candidates):
            raise ValueError("Candidate IDs must be unique within one query")
        import torch
        from lcm.candidate_transformer import CandidateTransformer

        # TreeCNN returns one tensor [D] for each candidate of this query.
        # The tree tensor adapter above remains pseudocode.
        E = torch.stack([self.encode_tree(p) for p in candidates], dim=0)
        if self.context_encoder is None:
            # Retain weights between calls; do not initialize a new block per query.
            self.context_encoder = CandidateTransformer(
                self.embedding_dim, self.attention_heads).to(device=E.device, dtype=E.dtype)
        # [N, D] -> [1, N, D]: attention runs ACROSS candidates, not tree nodes.
        # A production nn.Module adapter must register this block and set train/eval.
        H = self.context_encoder(E.unsqueeze(0)).squeeze(0)
        return [p.plan_id for p in candidates], H


class ProbabilisticPerformanceModel:
    """Pseudocode handoff: contextual plan embeddings -> performance parameters."""

    def __init__(self, encoder):
        self.encoder = encoder

    def forward(self, candidates):
        plan_ids, H = self.encoder.forward(candidates)
        Z = tensor_operation("SharedPredictionMLP", H)
        # Keep outputs aligned with plan_ids. Mean is on the normalized target
        # scale; inverse-transform before reporting actual latency units.
        mu = tensor_operation("SigmoidMeanHead", Z)
        variance = tensor_operation("SoftplusVarianceHead+epsilon", Z)
        # The Gaussian NLL interface uses its second channel as VARIANCE,
        # not standard deviation. No sqrt here.
        return plan_ids, mu, variance


class lcm_pl:
    """Compatibility sentinel for entry points that still import this name."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DURA now contains candidate-set encoder pseudocode. Implement "
            "grouped-query collation, TreeCNN tensor operations and the "
            "probabilistic training adapter before using legacy entry points. "
            "See lcm/candidate_transformer.py for the explicit attention implementation.")

    @classmethod
    def load_from_checkpoint(cls, *args, **kwargs):
        raise NotImplementedError("Old checkpoints are incompatible; retrain the new encoder.")
