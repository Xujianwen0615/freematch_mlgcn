import unittest

import torch
import torch.nn as nn

from data.rare_class_augmentation import RareClassBatchAugmenter
from mlgcn.freematch import FreeMatchMultiLabel
from mlgcn.losses import build_loss


class RareClassBatchAugmenterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        raw = torch.stack(
            [torch.full((3, 12, 10), fill_value=value) for value in (0.1, 0.3, 0.6, 0.9)]
        )
        self.mean = (0.2, 0.3, 0.4)
        self.std = (0.2, 0.25, 0.3)
        mean = torch.tensor(self.mean)[None, :, None, None]
        std = torch.tensor(self.std)[None, :, None, None]
        self.images = (raw - mean) / std

    def test_auto_selection_uses_labeled_count_quantile(self):
        augmenter = RareClassBatchAugmenter([100, 10, 20, 200], rare_quantile=0.25)
        self.assertEqual(augmenter.rare_class_ids, [1])

    def test_auto_selection_excludes_classes_without_labeled_anchors(self):
        augmenter = RareClassBatchAugmenter([100, 0, 10, 20], rare_quantile=0.25)
        self.assertEqual(augmenter.rare_class_ids, [2])

    def test_disabled_augmenter_is_an_identity(self):
        targets = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        )
        augmenter = RareClassBatchAugmenter([2, 2, 1], method="none")
        images, y_a, y_b, lam, loss_mask, stats = augmenter(self.images, targets)

        self.assertIs(images, self.images)
        self.assertIs(y_a, targets)
        self.assertIs(y_b, targets)
        self.assertTrue(torch.all(lam == 1.0))
        self.assertTrue(torch.all(loss_mask == 1.0))
        self.assertEqual(stats["augmented"], 0)

    def test_cutmix_returns_hard_target_pair_and_exact_lambda(self):
        targets = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        augmenter = RareClassBatchAugmenter(
            [2, 2, 1],
            method="cutmix",
            rare_class_ids=[0],
            probability=1.0,
            pairing="random",
            cutmix_alpha=1.0,
            cutmix_preserve_anchor=True,
            mean=self.mean,
            std=self.std,
        )

        images, y_a, y_b, lam, loss_mask, stats = augmenter(self.images, targets)

        self.assertEqual(images.shape, self.images.shape)
        self.assertTrue(torch.equal(y_a, targets))
        self.assertEqual(stats["eligible"], 2)
        self.assertEqual(stats["cutmix"], 2)
        self.assertTrue(torch.all(lam[:2] >= 0.5))
        self.assertTrue(torch.all(lam[:2] < 1.0))
        self.assertTrue(torch.all(lam[2:] == 1.0))
        self.assertTrue(torch.isfinite(images).all())
        self.assertTrue(torch.all((y_b == 0) | (y_b == 1)))
        self.assertTrue(torch.all(loss_mask == 1.0))
        self.assertFalse(torch.equal(images[:2], self.images[:2]))

    def test_ulda_label_subset_pairing_preserves_targets_and_pixel_range(self):
        targets = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        augmenter = RareClassBatchAugmenter(
            [2, 2, 1],
            method="ulda",
            rare_class_ids=[2],
            probability=1.0,
            ulda_operator="timsub",
            ulda_pairing="label_subset",
            mean=self.mean,
            std=self.std,
        )

        images, y_a, y_b, lam, loss_mask, stats = augmenter(self.images, targets)
        mean = torch.tensor(self.mean)[None, :, None, None]
        std = torch.tensor(self.std)[None, :, None, None]
        pixels = images * std + mean

        self.assertEqual(stats["timsub"], 2)
        self.assertTrue(torch.equal(y_a, targets))
        self.assertTrue(torch.equal(y_b, targets))
        self.assertTrue(torch.all(lam == 1.0))
        self.assertTrue(torch.all(loss_mask == 1.0))
        self.assertTrue(torch.isfinite(images).all())
        self.assertGreaterEqual(float(pixels.min()), -1e-6)
        self.assertLessEqual(float(pixels.max()), 1.0 + 1e-6)

    def test_ulda_skips_when_no_label_safe_donor_exists(self):
        targets = torch.eye(3)
        augmenter = RareClassBatchAugmenter(
            [1, 1, 1],
            method="ulda",
            rare_class_ids=[2],
            probability=1.0,
            ulda_pairing="label_subset",
            mean=self.mean,
            std=self.std,
        )
        images, y_a, _, _, loss_mask, stats = augmenter(self.images[:3], targets)
        self.assertEqual(stats["eligible"], 1)
        self.assertEqual(stats["augmented"], 0)
        self.assertTrue(torch.equal(images, self.images[:3]))
        self.assertTrue(torch.equal(y_a, targets))
        self.assertTrue(torch.all(loss_mask == 1.0))

    def test_ulda_masks_donor_only_positive_labels(self):
        targets = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        )
        augmenter = RareClassBatchAugmenter(
            [2, 1, 2],
            method="ulda",
            rare_class_ids=[2],
            probability=1.0,
            ulda_operator="timadd",
            ulda_pairing="shared_rare",
            mean=self.mean,
            std=self.std,
        )

        _, y_a, _, _, loss_mask, stats = augmenter(self.images, targets)

        self.assertEqual(stats["augmented"], 2)
        self.assertEqual(stats["ignored_labels"], 1)
        self.assertEqual(float(loss_mask[0, 0]), 0.0)
        self.assertTrue(torch.equal(y_a, targets))


class CutMixLossTests(unittest.TestCase):
    def test_asl_cutmix_is_weighted_sum_of_two_hard_target_losses(self):
        torch.manual_seed(11)
        model = nn.Linear(4, 3)
        criterion = build_loss("asl", reduction="none")
        consistency = build_loss("asl_soft", reduction="none")
        algorithm = FreeMatchMultiLabel(
            model,
            3,
            criterion,
            consistency,
            label_counts=[3, 2, 1],
            use_class_balancing=False,
            device="cpu",
            use_amp=False,
        )
        logits = torch.randn(2, 3, requires_grad=True)
        y_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_b = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        lam = torch.tensor([0.7, 0.55])

        actual = algorithm._sup_loss(logits, y_a, y_lb_b=y_b, mix_lam=lam)
        expected = (
            lam[:, None] * criterion(logits, y_a)
            + (1.0 - lam[:, None]) * criterion(logits, y_b)
        ).mean()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_supervised_mask_removes_ignored_entries_from_loss(self):
        model = nn.Linear(4, 3)
        criterion = build_loss("bce", reduction="none")
        algorithm = FreeMatchMultiLabel(
            model,
            3,
            criterion,
            criterion,
            loss_type="bce",
            label_counts=[1, 1, 1],
            use_class_balancing=False,
            device="cpu",
            use_amp=False,
        )
        logits = torch.tensor([[0.2, -0.4, 1.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0, 1.0]])

        actual = algorithm._sup_loss(logits, targets, sup_mask=mask)
        raw = criterion(logits, targets)
        expected = (raw * mask).sum() / mask.sum()
        self.assertTrue(torch.allclose(actual, expected))

    def test_public_supervised_step_accepts_cutmix_metadata(self):
        torch.manual_seed(13)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3))
        criterion = build_loss("asl", reduction="none")
        consistency = build_loss("asl_soft", reduction="none")
        algorithm = FreeMatchMultiLabel(
            model,
            3,
            criterion,
            consistency,
            label_counts=[3, 2, 1],
            device="cpu",
            use_amp=False,
        )
        algorithm.num_train_iter = 10
        x = torch.randn(2, 3, 4, 4)
        y_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_b = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        lam = torch.tensor([0.7, 0.55])

        log_dict, loss = algorithm.sup_train_step(
            x, y_a, y_lb_b=y_b, mix_lam=lam
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(log_dict["total_loss"], float(loss.item()), places=7)
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_semi_supervised_step_accepts_cutmix_metadata(self):
        torch.manual_seed(17)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3))
        criterion = build_loss("bce", reduction="none")
        algorithm = FreeMatchMultiLabel(
            model,
            3,
            criterion,
            criterion,
            loss_type="bce",
            label_counts=[3, 2, 1],
            device="cpu",
            use_amp=False,
        )
        algorithm.num_train_iter = 10
        x_lb = torch.randn(2, 3, 4, 4)
        y_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_b = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        lam = torch.tensor([0.7, 0.55])
        x_u_w = torch.randn(3, 3, 4, 4)
        x_u_s = torch.randn(3, 3, 4, 4)

        log_dict, loss = algorithm.train_step(
            x_lb, y_a, x_u_w, x_u_s, y_lb_b=y_b, mix_lam=lam
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("unsup_loss", log_dict)
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
