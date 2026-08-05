import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from failure_prob.data.utils import Rollout

from .conformal.functional_predictor import (
    FunctionalPredictor,
    ModulationType,
    RegressionType,
)

EVAL_TIMES = [
    "at earliest stop",
    "by earliest stop",
    "by final end",
]


def compute_roc(success_scores, fail_scores):
    y_true = [1] * len(fail_scores) + [0] * len(success_scores)
    y_score = fail_scores + success_scores
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return fpr, tpr, auc(fpr, tpr)


def compute_prc(success_scores, fail_scores):
    y_true = [1] * len(fail_scores) + [0] * len(success_scores)
    y_score = fail_scores + success_scores
    pre, rec, thresholds = precision_recall_curve(y_true, y_score)
    return pre, rec, auc(rec, pre)


def eval_binary_classification(scores, labels, threshold: float) -> dict:
    """Threshold-based binary classification metrics (failure = positive, 1)."""
    if isinstance(scores, list):
        scores = np.array(scores)
    if isinstance(labels, list):
        labels = np.array(labels)

    preds = (scores >= threshold).astype(int)
    labels = labels.astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    n = len(labels)
    acc = (tp + tn) / n if n > 0 else 0.0
    bal_acc = (tpr + tnr) / 2

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        roc_auc = float("nan")
        prc_auc = float("nan")
    else:
        roc_auc = roc_auc_score(labels, scores)
        prc_auc = average_precision_score(labels, scores)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "acc": acc,
        "bal_acc": bal_acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "prc_auc": prc_auc,
    }


def eval_fixed_threshold(
    rollouts_by_split_name: dict[str, list[Rollout]],
    scores_by_split_name: dict[str, list[np.ndarray]],
    method_name: str,
    thresholds: list[float] = [0.7],
) -> pd.DataFrame:
    classification_logs = []
    for split_name, rollouts in rollouts_by_split_name.items():
        scores_all = scores_by_split_name[split_name]
        labels = [1 - r.episode_success for r in rollouts]

        for eval_time in EVAL_TIMES:
            if eval_time == "at earliest stop":
                scores = [s[r.task_min_step - 1] for s, r in zip(scores_all, rollouts)]
            elif eval_time == "by earliest stop":
                scores = [s[: r.task_min_step].max() for s, r in zip(scores_all, rollouts)]
            elif eval_time == "by final end":
                scores = [s[: len(r.hidden_states)].max() for s, r in zip(scores_all, rollouts)]
            else:
                raise ValueError(f"Unknown eval_time: {eval_time}")

            for thresh in thresholds:
                result = eval_binary_classification(scores, labels, thresh)
                classification_logs.append(
                    {
                        "detect_method": method_name,
                        "split": split_name,
                        "thresh_method": "fixed",
                        "time": eval_time,
                        "threshold": thresh,
                        **result,
                    }
                )
    return pd.DataFrame(classification_logs)


def eval_functional_conformal(
    rollouts_by_split_name: dict[str, list[Rollout]],
    scores_by_split_name: dict[str, list[np.ndarray]],
    method_name: str,
    alphas: list[float],
    calib_split_names: list[str] = ["val_seen"],
    test_split_names: list[str] = ["val_unseen"],
    align_method: str = "extend",
) -> tuple[pd.DataFrame, dict]:
    classification_logs = []

    cal_rollouts, cal_scores_all = [], []
    for split_name in calib_split_names:
        cal_rollouts.extend(rollouts_by_split_name[split_name])
        cal_scores_all.extend(scores_by_split_name[split_name])

    test_rollouts, test_scores_all = [], []
    for split_name in test_split_names:
        test_rollouts.extend(rollouts_by_split_name[split_name])
        test_scores_all.extend(scores_by_split_name[split_name])
    test_labels_all = np.asarray([1 - r.episode_success for r in test_rollouts])

    test_earliest_stop = np.array([r.task_min_step for r in test_rollouts])
    if align_method == "extend":
        max_length = max(len(s) for s in cal_scores_all + test_scores_all)
        for i, s in enumerate(cal_scores_all):
            cal_scores_all[i] = np.pad(s, (0, max_length - len(s)), mode="edge")
        for i, s in enumerate(test_scores_all):
            test_scores_all[i] = np.pad(s, (0, max_length - len(s)), mode="edge")
    elif align_method == "truncate":
        raise NotImplementedError("Truncate alignment is not implemented yet.")
    else:
        raise ValueError(f"Unknown align_method: {align_method}")

    for calib_on in ["neg"]:
        if calib_on == "neg":
            lower_bound = False
            cal_scores_used = [s for s, r in zip(cal_scores_all, cal_rollouts) if r.episode_success == 1]
        else:
            raise NotImplementedError("Functional CP calibrated on failures does not make sense.")

        cal_scores_used = np.array(cal_scores_used)
        if len(cal_scores_used) == 0:
            raise RuntimeError("No successful calibration rollouts for functional CP.")
        if len(cal_scores_used) == 1:
            cal_scores_1 = cal_scores_used
            cal_scores_2 = cal_scores_used
        else:
            np.random.shuffle(cal_scores_used)
            n_cal_1 = int(len(cal_scores_used) * 0.3)
            cal_scores_1 = cal_scores_used[:n_cal_1]
            cal_scores_2 = cal_scores_used[n_cal_1:]
            # If the 30% regression split is empty (too few calibration scenes),
            # fall back to using the whole set for both roles.
            if len(cal_scores_1) == 0:
                cal_scores_1 = cal_scores_used
                cal_scores_2 = cal_scores_used

        test_scores_all = np.array(test_scores_all)
        n_test_samples = len(test_scores_all)
        cp_bands_by_alpha = {}

        for eval_time in ["by final end", "by earliest stop"]:
            for alpha in alphas:
                predictor = FunctionalPredictor(ModulationType.Tfunc, RegressionType.Mean)
                cp_band = predictor.get_one_sided_prediction_band(
                    cal_scores_1, cal_scores_2, alpha, lower_bound=lower_bound
                )
                cp_bands_by_alpha[alpha] = cp_band

                if lower_bound:
                    detection_mask = test_scores_all <= cp_band
                else:
                    detection_mask = test_scores_all >= cp_band

                if eval_time == "by final end":
                    lengths = test_scores_all.shape[1]
                elif eval_time == "by earliest stop":
                    lengths = test_earliest_stop
                    for i in range(len(test_scores_all)):
                        detection_mask[i, lengths[i]:] = False
                else:
                    raise ValueError(f"Unknown eval_time: {eval_time}")

                has_detection = np.any(detection_mask, axis=1)
                first_detection = np.argmax(detection_mask, axis=1)
                detection_times = np.where(has_detection, first_detection, lengths)
                relative_detection_times = detection_times / lengths

                pos_mask = test_labels_all == 1
                avg_det_time = np.mean(relative_detection_times[pos_mask]) if pos_mask.sum() > 0 else float("nan")
                predicted = has_detection
                tp = int((predicted & pos_mask).sum())
                fn = int((~predicted & pos_mask).sum())
                fp = int((predicted & ~pos_mask).sum())
                tn = int((~predicted & ~pos_mask).sum())

                with np.errstate(divide="ignore", invalid="ignore"):
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    acc = (tp + tn) / n_test_samples
                    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                    bal_acc = (tpr + tnr) / 2

                classification_logs.append(
                    {
                        "detect_method": method_name,
                        "cal split": "+".join(calib_split_names),
                        "test split": "+".join(test_split_names),
                        "calib on": calib_on,
                        "task": "all",
                        "thresh_method": "functional CP",
                        "alpha": alpha,
                        "time": eval_time,
                        "avg_det_time": avg_det_time,
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "tpr": tpr,
                        "tnr": tnr,
                        "fpr": fpr,
                        "fnr": fnr,
                        "acc": acc,
                        "bal_acc": bal_acc,
                        "f1": f1,
                    }
                )

    return pd.DataFrame(classification_logs), cp_bands_by_alpha


def eval_scores_roc_prc(
    rollouts_by_split_name: dict[str, list[Rollout]],
    scores_by_split_name: dict[str, list[np.ndarray]],
    method_name: str,
    time_quantiles: list[float],
) -> pd.DataFrame:
    """Return ROC / PRC AUC at the specified quantiles and early/end max scores."""
    records = []

    for split, rollouts in rollouts_by_split_name.items():
        scores = scores_by_split_name[split]
        labels = [1 - r.episode_success for r in rollouts]

        for tq in time_quantiles:
            fs = [s[round((r.task_min_step - 1) * tq)] for s, r in zip(scores, rollouts) if r.episode_success == 0]
            ss = [s[round((r.task_min_step - 1) * tq)] for s, r in zip(scores, rollouts) if r.episode_success == 1]
            if fs and ss:
                _, _, auc_tq = compute_roc(ss, fs)
                records.append({"split": split, "time": f"tq{tq}", "roc_auc": auc_tq})

        # binary classification AUC on full-trajectory max scores
        pos_scores = [s[: len(r.hidden_states)].max() for s, r in zip(scores, rollouts)]
        result = eval_binary_classification(pos_scores, labels, 0.5)
        records.append({"split": split, "time": "end_max", **result})

    return pd.DataFrame(records)
