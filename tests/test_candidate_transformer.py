import unittest
import torch
from lcm.candidate_transformer import CandidateTransformer
from lcm.dura_model import CandidatePlan, CandidateSetEncoder


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = CandidateTransformer(8, 2, dropout=0).eval()
        self.E = torch.randn(2, 3, 8)

    def test_candidate_permutation(self):
        order = [2, 0, 1]
        torch.testing.assert_close(self.model(self.E[:, order]), self.model(self.E)[:, order])

    def test_padding_and_query_isolation(self):
        mask = torch.tensor([[True, True, False], [True, True, True]])
        expected = self.model(self.E, mask)
        changed = self.E.clone()
        changed[0, 2] = 10000
        changed[1] = -400
        actual = self.model(changed, mask)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(expected[0, :2], self.model(self.E[0:1, :2])[0])
        self.assertEqual(actual[0, 2].abs().sum().item(), 0)

    def test_empty_mask_and_single_candidate(self):
        with self.assertRaises(ValueError):
            self.model(self.E, torch.zeros(2, 3, dtype=torch.bool))
        self.assertEqual(tuple(self.model(self.E[:, :1]).shape), (2, 1, 8))

    def test_gradients_and_context(self):
        E = self.E.clone().requires_grad_()
        self.model(E)[0, 0, 0].backward()
        self.assertTrue(torch.isfinite(E.grad).all())
        self.assertGreater(E.grad[0, 1:].abs().sum().item(), 0)
        self.assertEqual(E.grad[1].abs().sum().item(), 0)

    def test_tree_to_transformer_wiring(self):
        encoder = CandidateSetEncoder(2, 8, 2, context_encoder=self.model)
        # Isolate the implemented interface from the unfinished tree tensor adapter.
        encoder.encode_tree = lambda p: self.E[0, int(p.plan_id)]
        plans = [CandidatePlan('q1', str(i), [], []) for i in range(3)]
        ids, H = encoder.forward(plans)
        self.assertEqual(ids, ['0', '1', '2'])
        torch.testing.assert_close(H, self.model(self.E[:1])[0])


if __name__ == '__main__':
    unittest.main()
