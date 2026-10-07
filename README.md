# DURA

DURA explores candidate-set context-aware plan representations for learned query optimization. Its architecture combines TreeCNN structural embeddings with Transformer attention across candidate plans of the same query, followed by a probabilistic performance interface.

## Modules

| Directory | Purpose |
| --- | --- |
| `candidate_generation/` | Enumerates join orders, materializes and validates PostgreSQL plans, and converts candidates into model inputs. |
| `lcm/` | Defines plan features and model interfaces; implements cross-candidate Transformer attention. |
| `tcnn/` | Provides tree-convolution, normalization, activation, and pooling components. |
| `server/` | Contains the PostgreSQL 16.1 candidate-enumeration patch and corresponding source files. |
| `config/` | Stores candidate-generation settings and model design parameters. |
| `tests/` | Checks candidate generation, plan conversion, attention masks, and query isolation. |
| `docs/` | Describes setup and integration details. |

The representation flow is:

```text
SQL -> Candidate plans -> Node features -> TreeCNN -> Transformer -> Performance parameters
```

`lcm/candidate_transformer.py` contains the attention implementation. `lcm/dura_model.py` specifies the tree-encoding and prediction interfaces, including their implementation hooks.

## Usage

Use Python 3.10+, the supplied PostgreSQL 16.1 planner patch, and a compatible `pg_hint_plan`. Set `DURA_DATABASE_DSN` to your local database connection string.

```bash
python -m pip install -r requirements-candidates.txt
python -m candidate_generation --sql-file /path/to/query.sql --query-id q1
```

With PyTorch installed, run the tests:

```bash
python -m unittest discover -s tests -v
```

See [setup instructions](docs/candidate_generation.md) and [source references](THIRD_PARTY.md) for details.
