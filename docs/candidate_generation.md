# Diverse Candidate Plan Generation

This module includes an extracted PostgreSQL 16.1 join-order explorer and a Python adapter. The server changes retain additional candidate paths, collect final join orders, and export them. The client converts each order into a `pg_hint_plan` Leading hint, materializes it with `EXPLAIN (FORMAT JSON)`, checks the resulting join structure, and deduplicates plans.

This explores alternative join orders retained by the patched planner. It is not an exhaustive enumeration of every physical operator/scan combination, and it does not guarantee an optimal candidate. The candidate budget limits orders before plan deduplication, so fewer plans may be returned. Candidate generation does not use `EXPLAIN ANALYZE` or collect execution-time labels.

## Server component

`server/postgresql16_candidate_explorer.patch` targets PostgreSQL 16.1. `server/overlay/` contains the corresponding complete source files, including the enumeration code and planner/GUC/build integration. `server/base_manifest.json` records the expected base files. Apply the patch to a separate, clean PostgreSQL 16.1 source tree, then build and install that tree using the normal PostgreSQL build procedure:

```bash
cd /path/to/postgresql-16.1
patch --dry-run -p1 < /path/to/DURA/server/postgresql16_candidate_explorer.patch
patch -p1 < /path/to/DURA/server/postgresql16_candidate_explorer.patch
```

Use the GNU Make build path: the supplied integration updates the Makefiles, not Meson. Install a compatible `pg_hint_plan` into that server installation. Build validation here covers patch application and syntax checking of the exporter, not a complete server build or installation. PostgreSQL's copyright notice is retained in `server/COPYRIGHT`.

The server GUC is `enable_join_order_plans`. Each backend writes `/tmp/dura_join_order_plans_<backend_pid>.txt`; the Python client uses the matching backend PID. A previously installed server with another output filename will not work unchanged. Do not simply rename the client configuration and expect an unmodified server to match it.

The adapter must run on the database host, or both processes must see the same output path. The client needs permission to remove the stale file and read the newly generated file. Separate backends have separate paths; external code sharing one connection must not run concurrent planning operations on it.

## Generate candidates

```bash
python -m pip install -r requirements-candidates.txt
# Set DURA_DATABASE_DSN in your environment with your own local credentials.
python -m candidate_generation --sql-file /path/to/query.sql --query-id q1 \
  --schema public --config config/candidate_generation.json \
  --output-dir generated_plans
```

Use a trusted single SELECT statement. The default budget is 64. Output includes hinted SQL, join order, plan JSON, optimizer cost, and candidate ID. No credentials are bundled. The adapter may return an empty set for unsupported grouping constructs or rejected plans; callers must choose an explicit fallback policy.

## Connection to the DURA encoder

`candidate_generation.pipeline.generate_and_predict` performs:

```text
SQL -> enumerate candidates -> EXPLAIN and validate -> CandidatePlan objects
    -> ONE joint model.forward(candidates) call for the entire query
```

Provide a stable schema-level `relation_resolver` and a `scan_filter_resolver` based on query predicates and statistics. The latter must distinguish true filtering predicates from join-only index conditions. The adapter preserves tree topology and unions relation IDs at parent nodes. It rejects InitPlan/SubPlan nodes until an explicit subquery policy is implemented. Tree binarization and padding remain part of the encoder implementation work.

The generation component is concrete code. DURA's TreeCNN/attention/probabilistic model is still pseudocode and raises `NotImplementedError`; this integration does not turn it into a trained or runnable predictor. No new performance results have been measured.
