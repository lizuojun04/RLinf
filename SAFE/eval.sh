UV_PROJECT_ENVIRONMENT=.venv-safe uv run python eval_ckpt.py \
  --ckpt logs/safe_indep/seed_0/model_best.pt \
  --data-path ../data/libero \
  --task-ids 0 1 2 6 7 9 \
  --seen-train-ratio 0.75 \
  --unseen-task-ratio 0.33 \
  --max-per-class 20 \
  --eval-methods fixed functional roc \
  --cp-alphas 0.2 --fixed-thresholds "0.9" \
  --out-dir logs/eval_indep

# UV_PROJECT_ENVIRONMENT=.venv-safe uv run python eval_ckpt.py \
#   --ckpt logs/safe_lstm/seed_0/model_final.pt \
#   --data-path ../data/libero \
#   --task-ids 0 1 2 6 7 9 \
#   --seen-train-ratio 0.75 \
#   --unseen-task-ratio 0.33 --max-per-class 20 \
#   --eval-methods fixed functional roc \
#   --cp-alphas 0.2 --fixed-thresholds "0.5" \
#   --out-dir logs/eval_lstm
