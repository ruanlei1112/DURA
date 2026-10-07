# DURA

Candidate-set context-aware plan representation for probabilistic performance prediction.

**Status: implementation-oriented pseudocode, not a runnable training system.**
Only the DURA design is included. Legacy multi-model training scripts, benchmark
adapters, comparison notebooks and historical experiment tables are excluded.

## Architecture

```text
operator type + relation access + scan selectivity
    -> three shared TreeCNN layers and masked pooling per plan
    -> multi-head self-attention across candidates for ONE query
    -> contextual plan representations
    -> probabilistic mean and variance heads
```

## Files

- `lcm/dura_model.py`: node features, candidate-set contract and encoder/model pseudocode.
- `config/model_params.cfg`: DURA design parameters; not a ready training configuration.
- `docs/candidate_set_encoder.md`: feature definitions, grouping, masks and implementation requirements.
- `tcnn/tcnn.py`: shared tree-convolution reference utility, not yet wired into the pseudocode. Its original copyright and license notice are preserved.
- `THIRD_PARTY.md`: source references and attribution.

Use Python 3.10 or later for the pseudocode module. Importing it does not require
external packages. Abstract tensor operations intentionally raise
`NotImplementedError`. The optional tree-convolution reference requires PyTorch.
There is no bundled training or checkpoint-loading pipeline. Candidate generation and its server patch are now included; the prediction model remains pseudocode.

The source workspace is unchanged. No results for the proposed encoder have
been measured, and this package contains no performance claims or result tables.

## Candidate generation

See [candidate generation](docs/candidate_generation.md) for the PostgreSQL source patch, Python exporter, and connection to the candidate-set encoder. Install the separate `requirements-candidates.txt` only when using that component.
