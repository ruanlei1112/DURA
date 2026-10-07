# Source references

The earlier local research prototype was derived from the implementation
associated with [this research publication](https://arxiv.org/abs/2401.15210).
The present candidate-set encoder pseudocode follows the supplied PLARQ document
and the requested TreeCNN plus cross-candidate attention design. The new encoder
has not been trained or benchmarked.

The tree-convolution reference in `tcnn/tcnn.py` retains its original Ryan Marcus
copyright and GNU GPL notice. This packaging step does not grant a new license
for third-party code.

The candidate generator is extracted from the existing local PostgreSQL planner adaptation and its client routines. The server files retain PostgreSQL notices; see `server/COPYRIGHT`. This integration and neutral naming do not imply independent authorship of those components.
