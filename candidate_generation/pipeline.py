"""Connect generated plan documents to DURA's candidate-set model interface."""
from lcm.dura_model import CandidatePlan, PlanNode

SCAN_TYPES = {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan",
              "Bitmap Index Scan", "Parallel Seq Scan"}


def encode_candidates(query_id, records, relation_resolver, scan_filter_resolver):
    """Convert all generated plans for one query, preserving their order.

    relation_resolver(node) returns stable schema relation IDs directly accessed
    by a node (including index-to-table resolution for bitmap index scans).
    scan_filter_resolver(node) returns (has_filter, estimated_selectivity).
    It must use query predicates and statistics, never execution labels.
    Join-only index conditions are not treated as filtering predicates.
    """
    candidates = []
    for record in records:
        nodes, children = [], []

        def visit(node):
            index = len(nodes)
            nodes.append(None)
            children.append([])
            relations = set(relation_resolver(node))
            for child in node.get("Plans", []):
                if child.get("Parent Relationship") in {"InitPlan", "SubPlan"}:
                    raise NotImplementedError("Subquery plan encoding requires an explicit policy")
                child_index, child_relations = visit(child)
                children[index].append(child_index)
                relations.update(child_relations)
            is_scan = node.get("Node Type") in SCAN_TYPES
            has_filter, selectivity = scan_filter_resolver(node) if is_scan else (False, None)
            nodes[index] = PlanNode(node.get("Node Type", "Other"), sorted(relations),
                                    is_scan, has_filter, selectivity)
            return index, relations

        visit(record["plan_document"][0]["Plan"])
        candidates.append(CandidatePlan(str(query_id), str(record["candidate_id"]),
                                        nodes, children))
    return candidates


def generate_and_predict(schema, sql, query_id, model, relation_resolver,
                         scan_filter_resolver, **explorer_options):
    """Jointly predict a complete candidate set; the current model is pseudocode."""
    from .explorer import generate_candidate_plans
    records = generate_candidate_plans(schema, sql, query_id, **explorer_options)
    if not records:
        raise RuntimeError("No valid candidates; use an explicit baseline fallback policy")
    candidates = encode_candidates(query_id, records, relation_resolver, scan_filter_resolver)
    # One joint call, never one model call per candidate.
    return records, model.forward(candidates)
