# DURA Candidate-Set Context-Aware Encoder (Pseudocode)

This design follows the supplied Candidate-Set Context-Aware Plan Representation description and Section 4.1 of the PLARQ document. The previous query-graph GNN/MLP, table-node embedding augmentation, and global query-vector concatenation have been removed. This release contains only the DURA design and a tree-convolution reference utility, with no comparison models.

## Data flow

```text
Complete candidate set [P1, ..., Pn] for ONE query q
  -> Node features: 7 operator categories + relation multi-hot + scan selectivity
  -> Three shared TreeCNN layers per plan + masked dynamic pooling
  -> Structural embeddings E: [n, d]
  -> Cross-candidate multi-head self-attention + residual/LayerNorm + FFN
  -> Context-aware embeddings H: [n, d]
  -> Shared probabilistic performance model -> per-plan mean and variance
```

Relation vectors use a fixed schema vocabulary; nodes involving multiple relations may have multiple active entries. Selectivity is zero for non-scan nodes and scans without filtering predicates. Filtered scans use estimates from available statistics; LIKE predicates may use optimizer estimates. Test execution times, observed row counts, and other target labels must not enter the input features.

Residual connections, LayerNorm, and the feed-forward block are explicit design choices for this Transformer-based variant. MHSA is the core cross-candidate interaction in the supplied equations. No candidate positional embeddings are used, so deterministic inference is permutation-equivariant. Single-candidate sets are valid; empty sets are rejected.

## Required implementation work

1. Group input by `(workload, split, fold, seed, query_id)`. Do not split one query's candidates into independent random training examples. Attention must never mix different queries or training/validation/test partitions. Preserve alignment between candidate IDs and labels.
2. Re-extract node features with dimension `7 + relation_count + 1`, preserving the actual tree topology. Use explicit masks for missing children and variable-size trees. Operators with more than two children require a deterministic, consistent binarization policy. Old feature caches cannot be reused directly.
3. Batched embeddings have shape `[B, Cmax, d]`, with candidate padding masks of shape `[B, Cmax]`. Attention operates only along the candidate axis. Mask padded keys, discard or zero padded outputs, and compute loss only for valid candidates with valid labels. Never pass an entirely padded query to attention.
4. Do not predict each candidate independently in the inference pipeline. Individual tree embeddings may be computed separately, but assemble the full E and apply attention jointly once per query before predicting and selecting plans. Independent single-plan model calls cannot provide candidate-set context.
5. Implement a Lightning adapter. Feed H into the probabilistic mean and variance heads. For Gaussian negative log-likelihood, interpret the second output channel as variance, not standard deviation. Keep unit conversion and target normalization consistent with the training data.
6. Retrain the model; old checkpoints are incompatible. Previous query-graph parameters have been removed from the DURA configuration. The new configuration contains design parameters and is not yet connected to a trainer.

## Status and validation

Node feature extraction and input-group validation are executable. TreeCNN, MHSA, and the probabilistic heads are expressed as abstract pseudocode operations that explicitly raise `NotImplementedError`. The legacy `lcm_pl` entry point also raises instead of silently executing the previous encoder. This is not a complete runnable model.

After implementation, verify candidate permutation equivariance, isolation between queries, padding invariance, single-candidate behavior, variable-size trees, gradient propagation, and exclusion of test labels from input features.

Historical result tables have been removed from this release. The new encoder has not been trained, executed, or benchmarked, and no performance results are provided.
