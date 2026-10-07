import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from candidate_generation.explorer import _normalize_join_order, _validate_hinted_plan
from candidate_generation.pipeline import encode_candidates


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'Node Type': 'Hash Join', 'Plans': [
            {'Node Type': 'Seq Scan', 'Alias': 'a', 'Relation Name': 'one'},
            {'Node Type': 'Seq Scan', 'Alias': 'b', 'Relation Name': 'two'}]}

    def test_hint_validation(self):
        _validate_hinted_plan('(a b)', [{'Plan': self.plan}])
        with self.assertRaises(RuntimeError):
            _validate_hinted_plan('(b a)', [{'Plan': self.plan}])

    def test_unsafe_hint_rejected(self):
        with self.assertRaises(ValueError):
            _normalize_join_order('(a b) */ SELECT 1')

    def test_generation_refreshes_output_and_exports_candidates(self):
        from candidate_generation.explorer import generate_candidate_plans
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            order_file = directory / 'orders_42.txt'
            order_file.write_text('(stale other)\n')
            config = directory / 'config.json'
            config.write_text(json.dumps({'join_order_file': str(directory / 'orders_{backend_pid}.txt')}))
            connection = MagicMock()
            connection.get_backend_pid.return_value = 42
            cursor = connection.cursor.return_value.__enter__.return_value
            calls = []

            def fetch():
                if not calls:
                    self.assertFalse(order_file.exists())
                    order_file.write_text('(a b)\n(a b)\n')
                calls.append(True)
                return ([{'Plan': dict(self.plan, **{'Total Cost': 12.0})}],)

            cursor.fetchone.side_effect = fetch
            with patch('candidate_generation.explorer.connect_to_db', return_value=(connection, connection)):
                records = generate_candidate_plans('public', 'SELECT 1', 'q1', conn_str='test',
                    config_path=config, opt_plan_path=directory / 'plans')
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['hint'], 'Leading((a b))')
            self.assertTrue(Path(records[0]['plan_path']).is_file())
            self.assertEqual(connection.rollback.call_count, 2)
            connection.close.assert_called_once()

    def test_tree_and_query_alignment(self):
        records = [{'candidate_id': 3, 'plan_document': [{'Plan': self.plan}]}]
        mapping = {'one': 0, 'two': 1}
        candidates = encode_candidates('q1', records,
            lambda n: [mapping[n['Relation Name']]] if 'Relation Name' in n else [],
            lambda n: (True, 0.2) if n.get('Alias') == 'a' else (False, None))
        p = candidates[0]
        self.assertEqual((p.query_id, p.plan_id), ('q1', '3'))
        self.assertEqual(p.children, [[1, 2], [], []])
        self.assertEqual(p.nodes[0].relation_ids, [0, 1])
        self.assertEqual(p.nodes[1].scan_selectivity, 0.2)
        self.assertFalse(p.nodes[2].has_filter)


if __name__ == '__main__':
    unittest.main()
