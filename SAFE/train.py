"""Train a SAFE failure-detector on RLinf dump data (no wandb, no hydra).

Example:
    python train.py --data-path data/libero --model indep \
        --n-epochs 500 --batch-size 512 --lr 1e-3 \
        --seen-train-ratio 0.75 --unseen-task-ratio 0.33 \
        --max-per-class 20 --out-dir logs/safe_indep \
        --eval-methods fixed functional roc
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from failure_prob.data import (
    load_rollouts,
    normalize_rollouts_hidden_states,
    split_rollouts_by_seen_unseen,
)
from failure_prob.data.utils import RolloutDataset
from failure_prob.model import get_model
from failure_prob.utils.eval import score_rollouts
from failure_prob.utils.metrics import (
    eval_fixed_threshold,
    eval_functional_conformal,
    eval_scores_roc_prc,
)
from failure_prob.utils.random import seed_everything

DEFAULT_INDEP_CFG = {
    "n_layers": 2,
    "hidden_dim": 256,
    "final_act_layer": "sigmoid",
    "n_history_steps": 1,
    "use_threshold": False,
    "threshold": 50.0,
    "cumsum": True,
    "rmean": False,
    "use_time_weighting": False,
    "lambda_reg": 1.0,
}

DEFAULT_LSTM_CFG = {
    "n_layers": 1,
    "hidden_dim": 256,
    "n_history_steps": -1,
    "one_loss_per_seq": False,
    "lambda_reg": 1.0,
    "lambda_hard_neg": 0.0,
    "hard_neg_margin": 0.1,
    "hard_neg_beta": 50.0,
    "cumsum": False,
    "rmean": False,
    "use_time_weighting": False,
    "dropout": 0.0,
    "init_weight_scale": 1.0,
}


def build_cfg(args, model_name: str) -> dict:
    base = dict(DEFAULT_INDEP_CFG if model_name in ("indep", "mlp") else DEFAULT_LSTM_CFG)
    base.update(
        {
            "name": "indep" if model_name in ("indep", "mlp") else "lstm",
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "grad_max_norm": args.grad_max_norm,
        }
    )
    if args.hidden_dim is not None:
        base["hidden_dim"] = args.hidden_dim
    if args.n_layers is not None:
        base["n_layers"] = args.n_layers
    return base


def main():
    parser = argparse.ArgumentParser(description="Train a SAFE failure detector.")
    parser.add_argument("--data-path", required=True, help="Root dir with **/{success,fail}/*.pkl")
    parser.add_argument("--model", choices=["indep", "mlp", "lstm"], default="indep")
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd", "sgdm"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-step-size", type=int, default=300)
    parser.add_argument("--lr-gamma", type=float, default=1.0)
    parser.add_argument("--grad-max-norm", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--seen-train-ratio", type=float, default=0.6,
                        help="Fraction of each seen task used for train (rest = val_seen).")
    parser.add_argument("--unseen-task-ratio", type=float, default=0.3,
                        help="Fraction of tasks held out as val_unseen.")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap per (task, class) rollouts to this number (balanced sampling).")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="Only load these task IDs (e.g. '0 1 2 6 7 9'). Default: all.")
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument("--horizon-idx-rel", default="mean")
    parser.add_argument("--diff-idx-rel", default="mean")
    parser.add_argument("--normalize", action="store_true", help="Global zero-mean / unit-var normalization.")
    parser.add_argument("--seeds", default="0", help="e.g. '0' or '0-1-2'")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-methods", nargs="+", default=["fixed", "functional", "roc"],
                        choices=["fixed", "functional", "roc"])
    parser.add_argument("--cp-alphas", default="0.2", help="Comma-separated alphas for functional CP.")
    parser.add_argument("--fixed-thresholds", default="0.5",
                        help="Comma-separated thresholds for fixed-threshold eval (e.g. '0.3,0.5,0.7').")
    parser.add_argument("--roc-every", type=int, default=50, help="Run eval every N epochs.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split("-")]
    alphas = [float(x) for x in args.cp_alphas.split(",")]
    fixed_thresholds = [float(x) for x in args.fixed_thresholds.split(",")] if args.fixed_thresholds else []

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
    print(f"Loaded {len(all_rollouts)} rollouts.")

    for seed in seeds:
        cfg_out = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(cfg_out, exist_ok=True)
        print(f"\n=== Running seed {seed} ===")
        seed_everything(seed)

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

        train_rollouts = splits["train"]
        input_dim = train_rollouts[0].hidden_states.shape[-1]
        print(f"hidden feature dim: {input_dim}, sample shape {train_rollouts[0].hidden_states.shape}")

        model_cfg = build_cfg(args, args.model)
        model = get_model(args.model, input_dim, model_cfg)
        model.to(args.device)

        dataset_train = RolloutDataset(train_rollouts)
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
        optimizer, lr_scheduler = model.get_optimizer()

        best_metric: float = -1.0
        best_epoch: int = 0

        for epoch in range(model_cfg["n_epochs"]):
            model.train()
            model.train_epoch(optimizer, loader_train)
            if lr_scheduler is not None:
                lr_scheduler.step()

            if (epoch + 1) % args.roc_every == 0 or (epoch + 1) == model_cfg["n_epochs"]:
                print('-' * 50)
                scores = score_rollouts(model, splits, args.batch_size)
                calib_seed = seed * 1000 + (epoch + 1)
                metric = _evaluate_and_save(
                    args, model, splits, scores, alphas, fixed_thresholds, cfg_out,
                    tag=f"ep{epoch + 1}", calib_seed=calib_seed,
                )
                if metric is not None and metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch + 1
                    torch.save(
                        {
                            "model_name": args.model,
                            "input_dim": input_dim,
                            "model_cfg": model_cfg,
                            "state_dict": model.state_dict(),
                            "seed": seed,
                            "best_epoch": epoch + 1,
                            "calib_seed": calib_seed,
                        },
                        os.path.join(cfg_out, "model_best.pt"),
                    )
                    print(f"  [seed {seed}] new best (bal_acc={metric:.4f}) at epoch {best_epoch} (calib_seed={calib_seed})")

        # Save the final checkpoint
        ckpt_path = os.path.join(cfg_out, "model_final.pt")
        torch.save(
            {
                "model_name": args.model,
                "input_dim": input_dim,
                "model_cfg": model_cfg,
                "state_dict": model.state_dict(),
                "seed": seed,
                "best_epoch": best_epoch,
                "calib_seed": best_epoch * 1000 + seed if best_epoch > 0 else seed * 1000 + model_cfg["n_epochs"],
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")
        if best_metric >= 0:
            print(f"[seed {seed}] Best val_unseen bal_acc={best_metric:.4f} at epoch {best_epoch} -> model_best.pt")

        # Record split sizes for traceability
        with open(os.path.join(cfg_out, "splits.json"), "w") as f:
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


def _evaluate_and_save(args, model, splits, scores, alphas, fixed_thresholds, out_dir,
                       tag="final", calib_seed: int | None = None) -> float | None:
    """Run selected eval methods, write CSVs + terminal summary.

    Args:
        calib_seed: seed for the functional-CP 30/70 calibration split (see
            ``eval_functional_conformal``). Pass the same value used when
            reporting a metric so train/eval numbers match.

    Returns the primary metric (val_unseen functional-CP ``bal_acc`` averaged
    over alphas, or ``None`` if functional CP couldn't be computed), used by
    best-checkpoint selection.
    """
    has_seen = "val_seen" in splits and len(splits["val_seen"]) > 0
    has_unseen = "val_unseen" in splits and len(splits["val_unseen"]) > 0

    if "fixed" in args.eval_methods and fixed_thresholds:
        df = eval_fixed_threshold(splits, scores, "model", fixed_thresholds)
        df.to_csv(os.path.join(out_dir, f"fixed_threshold_{tag}.csv"), index=False)
        _print_fixed(df, fixed_thresholds, tag)

    best_bal_acc: float | None = None
    if "functional" in args.eval_methods and has_seen and has_unseen:
        df, _ = eval_functional_conformal(
            splits, scores, "model", alphas,
            calib_split_names=["val_seen"], test_split_names=["val_unseen"],
            calib_seed=calib_seed,
        )
        df.to_csv(os.path.join(out_dir, f"functional_cp_{tag}.csv"), index=False)
        _print_functional(df, alphas, tag)
        # Best metric = mean bal_acc over alphas at "by final end" on val_unseen.
        sub = df[df["time"] == "by final end"]
        if not sub.empty:
            best_bal_acc = float(sub["bal_acc"].mean())

    if "roc" in args.eval_methods:
        df = eval_scores_roc_prc(splits, scores, "model", [1.0])
        df.to_csv(os.path.join(out_dir, f"roc_auc_{tag}.csv"), index=False)
        _print_roc(df, tag)

    return best_bal_acc


def _print_fixed(df: pd.DataFrame, thresholds: list[float], tag: str):
    print(f"\n[fixed threshold {thresholds}] tag={tag}")
    for _, row in df.iterrows():
        if row["time"] != "by final end":
            continue
        if row["threshold"] not in thresholds:
            continue
        print(
            f"  {row['split']:10s} thresh={row['threshold']:<5} tp={row['tp']:3d} fp={row['fp']:3d} "
            f"tn={row['tn']:3d} fn={row['fn']:3d} "
            f"recall={row['tp'] / (row['tp'] + row['fn']):.3f} precision={row['tp'] / (row['tp'] + row['fp']):.3f} "
            f"acc={row['acc']:.3f} bal_acc={row['bal_acc']:.3f}"
        )


def _print_functional(df: pd.DataFrame, alphas, tag: str):
    print(f"\n[functional CP] tag={tag}")
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


def _print_roc(df: pd.DataFrame, tag: str):
    print(f"\n[ROC AUC] tag={tag}")
    for _, row in df.iterrows():
        if row["time"] == "end_max":
            print(f"  {row['split']:10s} end_max roc_auc={row['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
