import unittest

import torch

from src.models.daeac_adversarial import DAEACDANNModel
from src.models.daeac_paper import DAEACFeatureExtractor, LateFusionClassifierH
from src.training.train_daeac_adversarial import _target_batch_with_optional_rr


class DAEACDANNLateFusionTests(unittest.TestCase):
    def test_dann_late_fusion_forward_uses_rr_for_class_logits(self) -> None:
        feature_extractor = DAEACFeatureExtractor(
            input_channels=1,
            initial_channels=4,
            feature_dim=256,
            input_rows=1,
            attention_type="fcba",
        )
        classifier = LateFusionClassifierH(feature_dim=256, num_classes=3, rr_dim=7, fc1_dim=128, fc2_dim=64)
        model = DAEACDANNModel(
            feature_extractor=feature_extractor,
            classifier=classifier,
            feature_dim=128,
            num_classes=3,
            domain_hidden_dim=128,
        )

        x = torch.randn(4, 1, 1, 128)
        rr_features = torch.randn(4, 7)
        logits, features = model(x, rr_features=rr_features, return_embedding=True)
        domain_logits = model.forward_domain(x, lambd=0.5)

        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertEqual(tuple(features.shape), (4, 128))
        self.assertEqual(tuple(domain_logits.shape), (4, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(domain_logits).all())

    def test_target_batch_with_rr_uses_index_not_target_label(self) -> None:
        x = torch.randn(2, 1, 1, 128)
        rr_features = torch.randn(2, 7)
        index = torch.tensor([10, 11])

        moved_x, moved_rr = _target_batch_with_optional_rr((x, rr_features, index), torch.device("cpu"))

        self.assertTrue(torch.equal(moved_x, x))
        self.assertTrue(torch.equal(moved_rr, rr_features))


if __name__ == "__main__":
    unittest.main()
