import numpy as np

from scripts import eval_mse


def test_compute_action_mse_normalizes_gt_once():
    norm_stats = {
        "actions": {
            "mean": np.array([5.0, 20.0], dtype=np.float32),
            "std": np.array([1.0, 1.0], dtype=np.float32),
            "q01": np.array([0.0, 10.0], dtype=np.float32),
            "q99": np.array([10.0, 30.0], dtype=np.float32),
        }
    }
    pred_norm = np.array([[[0.0, 1.0]]], dtype=np.float32)
    gt_raw = np.array([[[5.0, 30.0]]], dtype=np.float32)

    metrics = eval_mse.compute_action_mse(pred_norm, gt_raw, norm_stats, action_dims=2)

    np.testing.assert_allclose(metrics["mse_total_norm"], 0.0, atol=1e-10)
    np.testing.assert_allclose(metrics["mse_total_phys"], 0.0, atol=1e-10)
    np.testing.assert_allclose(metrics["gt_norm"][0, 0], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(metrics["gt_phys"][0, 0], gt_raw[0, 0], atol=1e-6)
