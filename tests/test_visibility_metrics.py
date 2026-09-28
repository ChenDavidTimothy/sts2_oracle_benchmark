import math

import torch

from sts2.metrics import depth_agreement, visibility_metrics, radiance_hit_mask, psnr


def test_missing_and_wrong_first_hits_are_counted():
    ref_depth = torch.tensor([[1.0, 2.0, 3.0, float("inf")]])
    pred_depth = torch.tensor([[1.0, float("inf"), 3.2, 4.0]])
    ref_id = torch.tensor([[0, 1, 1, -1]])
    pred_id = torch.tensor([[0, -1, 0, 0]])
    mask = torch.ones_like(ref_depth, dtype=torch.bool)
    m = visibility_metrics(pred_depth, ref_depth, pred_id, ref_id, mask, tol=0.05)
    assert m["hit_precision"] == 2 / 3
    assert m["hit_recall"] == 2 / 3
    assert m["hit_f1"] == 2 / 3
    assert math.isclose(m["matched_depth_rmse"], math.sqrt(0.02), rel_tol=1e-5)
    assert math.isclose(m["matched_depth_max_error"], 0.2, rel_tol=1e-5)
    assert m["first_hit_surface_agreement"] == 0.5
    assert m["first_hit_surface_recall"] == 1 / 3
    assert math.isclose(depth_agreement(pred_depth, ref_depth, mask, tol=0.05), 1 / 3, rel_tol=1e-6)


def test_radiance_and_union_masks_expose_false_coverage():
    roles = torch.tensor([1, 0], dtype=torch.int32)
    reference_ids = torch.tensor([[0, -1, 1, -1]])
    predicted_ids = torch.tensor([[0, 0, 1, -1]])
    ref_hit = radiance_hit_mask(reference_ids, roles)
    pred_hit = radiance_hit_mask(predicted_ids, roles)
    assert ref_hit.tolist() == [[True, False, False, False]]
    assert (ref_hit | pred_hit).tolist() == [[True, True, False, False]]
    ref = torch.zeros((1, 4, 3))
    pred = ref.clone()
    pred[0, 1] = 1.0
    assert psnr(pred, ref, ref_hit) == 99.0
    assert psnr(pred, ref, ref_hit | pred_hit) < 10.0
