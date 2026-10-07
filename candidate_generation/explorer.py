"""Generate join-order candidates and validate materialized PostgreSQL plans.

Requires the bundled PostgreSQL 16.1 server patch and pg_hint_plan.
Uses EXPLAIN without ANALYZE; it never labels candidates by executing queries.
"""
import json
import os
import re
import threading
from pathlib import Path
import psycopg2
from psycopg2 import sql as pg_sql

_generation_lock = threading.Lock()
_last_explain_result = {}
_DEFAULT_CONFIG = {
    "enabled": True,
    "connection_string": None,
    "join_order_file": "/tmp/dura_join_order_plans_{backend_pid}.txt",
    "max_candidate_plans": 64,
    "deduplicate_plans": True,
    "load_pg_hint_plan": True,
}


def connect_to_db(conn_str):
    connection = psycopg2.connect(conn_str)
    return connection, connection


def close_connection_to_db(connection):
    connection.close()

def load_config(config_path=None):
    """Load dura_PG settings, with safe defaults for local deployments."""
    if config_path is None:
        config_path = os.environ.get("DURA_EXPLORER_CONFIG")
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "candidate_generation.json"

    config = dict(_DEFAULT_CONFIG)
    config_path = Path(config_path)
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as config_file:
            user_config = json.load(config_file)
        if not isinstance(user_config, dict):
            raise ValueError("dura_PG config must be a JSON object.")
        config.update(user_config)
    return config

def _normalize_join_order(join_order):
    join_order = " ".join(join_order.strip().split())
    if not join_order:
        return None
    # dura writes only aliases, parentheses and whitespace. Reject anything
    # else before embedding it in a pg_hint_plan comment.
    if not re.fullmatch(r"[()\sA-Za-z0-9_$]+", join_order):
        raise ValueError("Unsafe dura join order: {!r}".format(join_order))
    if not re.search(r"[A-Za-z_][A-Za-z0-9_$]*", join_order):
        raise ValueError("dura join order has no table alias: {!r}".format(join_order))
    if join_order.count("(") != join_order.count(")"):
        raise ValueError("Unbalanced dura join order: {!r}".format(join_order))
    return join_order

def _hinted_sql(statement, join_order):
    return "/*+ Leading({}) */\n{}".format(join_order, statement)

def _parse_join_order(join_order):
    tokens = re.findall(r"\(|\)|[A-Za-z_][A-Za-z0-9_$]*", join_order)
    position = 0
    strictly_binary = True

    def parse_item():
        nonlocal position, strictly_binary
        if position >= len(tokens):
            raise ValueError("Incomplete dura join order.")
        token = tokens[position]
        position += 1
        if token != "(":
            if token == ")":
                raise ValueError("Unexpected ')' in dura join order.")
            return token.casefold()

        items = []
        while position < len(tokens) and tokens[position] != ")":
            items.append(parse_item())
        if position >= len(tokens):
            raise ValueError("Unbalanced dura join order.")
        position += 1
        if len(items) < 2:
            raise ValueError("dura join groups must contain at least two items.")
        if len(items) != 2:
            strictly_binary = False
        result = items[0]
        for item in items[1:]:
            result = (result, item)
        return result

    structure = parse_item()
    if position != len(tokens):
        raise ValueError("Unexpected trailing dura join-order tokens.")
    return structure, strictly_binary

def _plan_join_structure(node):
    child_structures = []
    for child in node.get("Plans", []):
        if child.get("Parent Relationship") in ("InitPlan", "SubPlan"):
            continue
        structure = _plan_join_structure(child)
        if structure is not None:
            child_structures.append(structure)

    node_type = node.get("Node Type")
    if node_type in (
        "Nested Loop",
        "NestLoop",
        "Hash Join",
        "HashJoin",
        "Merge Join",
        "MergeJoin",
    ):
        if len(child_structures) != 2:
            return None
        return child_structures[0], child_structures[1]
    if len(child_structures) == 1:
        return child_structures[0]
    if child_structures:
        return tuple(child_structures)

    relation_name = node.get("Alias") or node.get("Relation Name")
    return relation_name.casefold() if relation_name else None

def _flatten_join_structure(structure):
    if isinstance(structure, tuple):
        aliases = []
        for item in structure:
            aliases.extend(_flatten_join_structure(item))
        return aliases
    return [structure] if structure is not None else []

def _validate_hinted_plan(join_order, plan_document):
    expected, strictly_binary = _parse_join_order(join_order)
    actual = _plan_join_structure(plan_document[0]["Plan"])
    if _flatten_join_structure(expected) != _flatten_join_structure(actual):
        raise RuntimeError(
            "pg_hint_plan did not produce the dura join order {!r}.".format(
                join_order
            )
        )
    if strictly_binary and expected != actual:
        raise RuntimeError(
            "pg_hint_plan changed the dura join tree {!r}.".format(join_order)
        )

def generate_candidate_plans(
    schema_name,
    sql,
    query_id,
    conn_str=None,
    config_path=None,
    opt_plan_path="./optimizer_plans",
):
    # dura uses one server-side output file, so serialize generation within
    # this process to prevent concurrent queries from consuming each other's data.
    with _generation_lock:
        return _generate_candidate_plans(
            schema_name,
            sql,
            query_id,
            conn_str=conn_str,
            config_path=config_path,
            opt_plan_path=opt_plan_path,
        )

def _generate_candidate_plans(
    schema_name,
    sql,
    query_id,
    conn_str=None,
    config_path=None,
    opt_plan_path="./optimizer_plans",
):
    """Enumerate dura join orders and materialize their PostgreSQL plans.

    dura_PG writes join orders to ``join_order_file``.  Each order is fed
    back through pg_hint_plan as a Leading hint so that DURA receives actual,
    executable candidate plans rather than join-order strings only.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(query_id)):
        raise ValueError("query_id must be a safe file identifier")
    config = load_config(config_path)
    if not config["enabled"]:
        return []
    if conn_str is None:
        configured_conn_str = config.get("connection_string")
        if configured_conn_str and "CHANGE_ME" not in configured_conn_str:
            conn_str = configured_conn_str
    if conn_str is None:
        conn_str = os.environ.get("DURA_DATABASE_DSN")
    if not conn_str:
        raise ValueError("Supply conn_str or DURA_DATABASE_DSN")

    max_candidates = int(config["max_candidate_plans"])
    if max_candidates < 1:
        raise ValueError("max_candidate_plans must be at least 1.")

    statement = sql.replace("\n", " ").strip().rstrip(";")
    unsupported_grouping = re.search(
        r"\b(?:rollup|cube)\s*\(|\bgrouping\s+sets\s*\(",
        statement,
        flags=re.IGNORECASE,
    )
    if unsupported_grouping:
        # The bundled dura PG16 extension raises an internal
        # "unrecognized node type" error for grouping-set plan nodes.  Avoid
        # invoking it for every generated instance of DSB template 018; the
        # caller will retain and evaluate PostgreSQL's ordinary fallback plan.
        print(
            "Skipping dura exploration for query {}: the bundled dura "
            "extension does not support ROLLUP/CUBE/GROUPING SETS.".format(query_id),
            flush=True,
        )
        return []
    output_dir = Path(opt_plan_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    connection, _ = connect_to_db(conn_str)
    if connection is None:
        raise RuntimeError("Could not connect to dura_PG.")

    try:
        # Match the backend-specific output path in the bundled server patch.
        join_order_file = Path(config["join_order_file"].format(backend_pid=connection.get_backend_pid()))
        default_join_order_file = join_order_file
        join_order_files = [join_order_file]
        _last_explain_result["plan_cost"] = None
        _last_explain_result["plan_document"] = None
        _last_explain_result["schema_name"] = None
        # Remove every path we may consume.  Therefore a fallback can never
        # silently reuse join orders left by a previous query or experiment.
        for candidate_file in join_order_files:
            if candidate_file.is_file():
                try:
                    candidate_file.unlink()
                except OSError as exc:
                    raise RuntimeError(
                        "Could not remove the stale dura join-order file {!s}: {}".format(
                            candidate_file, exc
                        )
                    ) from exc

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    pg_sql.SQL("SET LOCAL search_path TO {}, public").format(
                        pg_sql.Identifier(schema_name)
                    )
                )
                if config.get("load_pg_hint_plan", True):
                    cursor.execute("LOAD 'pg_hint_plan'")
                    cursor.execute("SET LOCAL pg_hint_plan.message_level = 'error'")
                cursor.execute("SET LOCAL enable_join_order_plans = on")
                cursor.execute(
                    pg_sql.SQL("EXPLAIN (FORMAT JSON) {}").format(
                        pg_sql.SQL(statement)
                    )
                )
                cursor.fetchone()
        except psycopg2.InternalError as exc:
            # dura's PostgreSQL extension does not handle every PG16 plan
            # node (DSB GROUP BY ROLLUP currently raises "unrecognized node
            # type").  This is a query-local explorer limitation: retain the
            # ordinary PostgreSQL plan as the safe fallback and continue the
            # fold instead of discarding hours of completed work.
            connection.rollback()
            print(
                "Skipping dura exploration for query {} because the dura "
                "extension rejected this PostgreSQL plan: {}".format(
                    query_id, " ".join(str(exc).split())
                ),
                flush=True,
            )
            return []
        connection.rollback()

        generated_files = [path for path in join_order_files if path.is_file()]
        if not generated_files:
            raise FileNotFoundError(
                "dura join-order file was not found at any expected path: {}. The file is "
                "created on the PostgreSQL server, so DURA must run on that "
                "host or the configured path must be shared. The dura build in this "
                "repository writes to {!s}.".format(
                    ", ".join(str(path) for path in join_order_files),
                    default_join_order_file,
                )
            )
        # Prefer the configured path when the server supports it; otherwise
        # consume the repository extension's fixed default path.
        join_order_file = generated_files[0]

        join_orders = []
        seen_orders = set()
        with join_order_file.open("r", encoding="utf-8") as order_file:
            for raw_order in order_file:
                order = _normalize_join_order(raw_order)
                if order and order not in seen_orders:
                    seen_orders.add(order)
                    join_orders.append(order)
                if len(join_orders) >= max_candidates:
                    break

        candidates = []
        seen_plans = set()
        for candidate_id, join_order in enumerate(join_orders):
            hinted_statement = _hinted_sql(statement, join_order)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        pg_sql.SQL("SET LOCAL search_path TO {}, public").format(
                            pg_sql.Identifier(schema_name)
                        )
                    )
                    cursor.execute("SET LOCAL pg_hint_plan.message_level = 'error'")
                    cursor.execute("SET LOCAL enable_join_order_plans = off")
                    cursor.execute(
                        pg_sql.SQL("EXPLAIN (FORMAT JSON) {}").format(
                            pg_sql.SQL(hinted_statement)
                        )
                    )
                    explain_value = cursor.fetchone()[0]
            except psycopg2.Error as exc:
                # One malformed/unsupported dura join order must not abort
                # candidate generation for the remaining valid orders.
                connection.rollback()
                print(
                    "Skipping dura candidate {} for query {} because "
                    "PostgreSQL rejected it: {}".format(
                        candidate_id, query_id, " ".join(str(exc).split())
                    ),
                    flush=True,
                )
                continue
            connection.rollback()
            plan_document = (
                explain_value
                if isinstance(explain_value, list)
                else json.loads(explain_value)
            )
            try:
                _validate_hinted_plan(join_order, plan_document)
            except RuntimeError as exc:
                print(
                    "Skipping dura candidate {} for query {}: {}".format(
                        candidate_id,
                        query_id,
                        exc,
                    )
                )
                continue
            plan_signature = json.dumps(
                plan_document[0]["Plan"], sort_keys=True, separators=(",", ":")
            )
            if config.get("deduplicate_plans", True) and plan_signature in seen_plans:
                continue
            seen_plans.add(plan_signature)

            plan_path = output_dir / "query#{}-dura-{}.json".format(
                query_id, candidate_id
            )
            with plan_path.open("w", encoding="utf-8") as plan_file:
                json.dump(plan_document, plan_file, indent=2)
            _last_explain_result["plan_document"] = plan_document
            _last_explain_result["plan_cost"] = plan_document[0]["Plan"]["Total Cost"]
            _last_explain_result["schema_name"] = schema_name
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "join_order": join_order,
                    "hint": "Leading({})".format(join_order),
                    "sql": hinted_statement,
                    "plan_path": str(plan_path),
                    "plan_cost": plan_document[0]["Plan"]["Total Cost"],
                    "plan_document": plan_document,
                }
            )
        return candidates
    finally:
        close_connection_to_db(connection)
