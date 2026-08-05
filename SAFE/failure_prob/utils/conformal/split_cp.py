import numpy as np
import torch


def quantile_threshold(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """ceil((N+1)*(1-alpha))-th smallest value of the provided scores."""
    N = scores.numel()
    k = int(torch.ceil(torch.tensor((N + 1) * (1 - alpha), dtype=torch.float)))
    k = np.clip(k, 1, N)
    sorted_scores, _ = torch.sort(scores)
    return sorted_scores[k - 1]


def split_conformal_binary(cal_scores, cal_labels, test_scores, alpha):
    """Performs split conformal prediction for binary classification.

    Nonconformity: label 1 -> 1 - s, label 0 -> s. Returns prediction sets and
    per-class thresholds dict keyed by candidate label {0, 1}.
    """
    if isinstance(cal_scores, list):
        cal_scores = torch.tensor(cal_scores)
    if isinstance(cal_labels, list):
        cal_labels = torch.tensor(cal_labels)
    if isinstance(test_scores, list):
        test_scores = torch.tensor(test_scores)

    thresholds = {}

    pos_mask = cal_labels == 1
    if pos_mask.sum() > 0:
        cal_pos_nconf = 1 - cal_scores[pos_mask]
        thresholds[1] = quantile_threshold(cal_pos_nconf, alpha).item()
    else:
        thresholds[1] = float("inf")

    neg_mask = cal_labels == 0
    if neg_mask.sum() > 0:
        cal_neg_nconf = cal_scores[neg_mask]
        thresholds[0] = quantile_threshold(cal_neg_nconf, alpha).item()
    else:
        thresholds[0] = float("inf")

    prediction_sets = []
    for s in test_scores:
        pred_set = set()
        if (1 - s) <= thresholds[1]:
            pred_set.add(1)
        if s <= thresholds[0]:
            pred_set.add(0)
        prediction_sets.append(pred_set)

    return prediction_sets, thresholds
