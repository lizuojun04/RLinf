"""Evaluate a trained SAFE checkpoint on a dump root (no wandb).

Loads the ckpt saved by ``train.py``, scores the rollouts, and exports the
full metric tables (tp/fp/tn/fn, accuracy, balanced accuracy, ROC AUC, and
optional functional / split conformal prediction).

Example:
    python eval_ckpt.py --ckpt logs/safe_indep/seed_0/model_final.pt \
        --data-path data/libero --seen-train-ratio 0.75 --unseen-task-ratio 0.33 \
        --max-per-class 20 --subsample-seed 0 --out-dir logs/eval_indep
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from failure_prob.data import (
    load_rollouts,
    normalize_rollouts_hidden_states,
    split_rollouts_by_seen_unseen,
)
from failure_prob.model import get_model
from failure_prob.utils.eval import score_rollouts
from failure_prob.utils.metrics import (
    eval_fixed_threshold,
    eval_functional_conformal,
    eval_scores_roc_prc,
)
from failure_prob.utils.random import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAFE checkpoint.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--seen-train-ratio", type=float, default=0.6)
    parser.add_argument("--unseen-task-ratio", type=float, default=0.3)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="Only load these task IDs. Default: all.")
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument("--horizon-idx-rel", default="mean")
    parser.add_argument("--diff-idx-rel", default="mean")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-methods", nargs="+", default=["fixed", "functional", "roc"],
                        choices=["fixed", "functional", "roc"])
    parser.add_argument("--cp-alphas", default="0.2")
    parser.add_argument("--fixed-thresholds", default="0.5",
                        help="Comma-separated thresholds for fixed-threshold eval (e.g. '0.3,0.5,0.7').")
    parser.add_argument("--calib-seed", type=int, default=None,
                        help="Seed for the functional-CP 30/70 calibration split. "
                             "Default: derive from the ckpt's stored ``best_epoch``/``seed`` "
                             "(i.e. ``seed * 1000 + best_epoch``) so train/eval match exactly. "
                             "Set explicitly to override.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    alphas = [float(x) for x in args.cp_alphas.split(",")]
    fixed_thresholds = [float(x) for x in args.fixed_thresholds.split(",")] if args.fixed_thresholds else []

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model_name = ckpt["model_name"]
    input_dim = ckpt["input_dim"]
    model_cfg = ckpt["model_cfg"]
    model = get_model(model_name, input_dim, model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    print(f"Loaded {model_name} ckpt (input_dim={input_dim}) from {args.ckpt}")

    # Derive the functional-CP calibration seed so train/eval use the identical
    # 30/70 split of the val_seen success scores. train.py stores it as
    # ``seed * 1000 + best_epoch``; fall back to a bare ckpt-derived value for
    # checkpoints saved without the field. An explicit --calib-seed overrides.
    calib_seed = args.calib_seed
    if calib_seed is None:
        ckpt_seed = ckpt.get("seed", 0)
        ckpt_epoch = ckpt.get("best_epoch")
        if ckpt_epoch is not None:
            calib_seed = int(ckpt_seed) * 1000 + int(ckpt_epoch)
        elif "calib_seed" in ckpt:
            calib_seed = int(ckpt["calib_seed"])
        else:
            print("WARNING: ckpt has no best_epoch/calib_seed and --calib-seed not set; "
                  "functional CP will use a non-deterministic split.")
    print(f"Functional-CP calib_seed: {calib_seed}")

    seed_everything(0)
    all_rollouts = load_rollouts(
        args.data_path,
        horizon_idx_rel=args.horizon_idx_rel,
        diff_idx_rel=args.diff_idx_rel,
        task_ids=args.task_ids,
        max_per_class_per_task=args.max_per_class,
        seed=args.subsample_seed,
        load_to_cuda=False,
    )
    if args.normalize:
        all_rollouts = normalize_rollouts_hidden_states(all_rollouts)

    task_ids = sorted({r.task_id for r in all_rollouts})
    n_unseen = max(round(args.unseen_task_ratio * len(task_ids)), 1)
    n_seen = len(task_ids) - n_unseen
    np.random.shuffle(task_ids)
    seen_task_ids = task_ids[:n_seen]
    unseen_task_ids = task_ids[n_seen:]
    print(f"Task split -> seen: {seen_task_ids}, unseen: {unseen_task_ids}")

    splits = split_rollouts_by_seen_unseen(
        all_rollouts, seen_task_ids, unseen_task_ids, seen_train_ratio=args.seen_train_ratio
    )

    scores = score_rollouts(model, splits, args.batch_size)

    if "fixed" in args.eval_methods and fixed_thresholds:
        df = eval_fixed_threshold(splits, scores, "model", fixed_thresholds)
        df.to_csv(os.path.join(args.out_dir, "fixed_threshold.csv"), index=False)
        print(f"\n[fixed threshold {fixed_thresholds}] by final end")
        for _, row in df.iterrows():
            if row["threshold"] in fixed_thresholds and row["time"] == "by final end":
                print(
                    f"  {row['split']:10s} thresh={row['threshold']:<5} tp={row['tp']:3d} fp={row['fp']:3d} "
                    f"tn={row['tn']:3d} fn={row['fn']:3d} "
                    f"recall={row['tp'] / (row['tp'] + row['fn']):.3f} precision={row['tp'] / (row['tp'] + row['fp']):.3f} "
                    f"acc={row['acc']:.3f} bal_acc={row['bal_acc']:.3f}"
                )

    if "functional" in args.eval_methods and splits.get("val_seen") and splits.get("val_unseen"):
        df, cp_bands = eval_functional_conformal(
            splits, scores, "model", alphas,
            calib_split_names=["val_seen"], test_split_names=["val_unseen"],
            calib_seed=calib_seed,
        )
        df.to_csv(os.path.join(args.out_dir, "functional_cp.csv"), index=False)
        print("\n[functional CP, test=val_unseen] by final end")
        for alpha in alphas:
            sub = df[(df["alpha"] == alpha) & (df["time"] == "by final end")]
            if sub.empty:
                continue
            row = sub.iloc[0]
            print(
                f"  alpha={alpha} tp={row['tp']:3d} fp={row['fp']:3d} tn={row['tn']:3d} fn={row['fn']:3d} "
                f"recall={row['tp'] / (row['tp'] + row['fn']):.3f} precision={row['tp'] / (row['tp'] + row['fp']):.3f} "
                f"acc={row['acc']:.3f} bal_acc={row['bal_acc']:.3f}"
            )
        np.savez(os.path.join(args.out_dir, "cp_bands.npz"), **{f"a{alpha}": cp_bands[alpha] for alpha in cp_bands})

    if "roc" in args.eval_methods:
        df = eval_scores_roc_prc(splits, scores, "model", [1.0])
        df.to_csv(os.path.join(args.out_dir, "roc_auc.csv"), index=False)
        print("\n[ROC AUC] end_max")
        for _, row in df.iterrows():
            if row["time"] == "end_max":
                print(f"  {row['split']:10s} roc_auc={row['roc_auc']:.4f} (tpr={row['tpr']:.3f} tnr={row['tnr']:.3f})")

    with open(os.path.join(args.out_dir, "split_info.json"), "w") as f:
        json.dump(
            {
                "seen_task_ids": seen_task_ids,
                "unseen_task_ids": unseen_task_ids,
                "n_train": len(splits["train"]),
                "n_val_seen": len(splits.get("val_seen", [])),
                "n_val_unseen": len(splits.get("val_unseen", [])),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
