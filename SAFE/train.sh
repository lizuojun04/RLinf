# seen:unseen = (1 - unseen-task-ratio) : unseen-task-ratio = 0.67 : 0.33
# seen_train:seen_eval = seen-train-ratio : (1 - seen-train-ratio) = 0.75 : 0.25

UV_PROJECT_ENVIRONMENT=.venv-safe uv run python train.py \
  --data-path ../data/libero \
  --task-ids 0 1 2 6 7 9 \
  --model indep \
  --n-epochs 250 \
  --seen-train-ratio 0.75 \
  --unseen-task-ratio 0.33 \
  --max-per-class 20 \
  --out-dir logs/safe_indep \
  --eval-methods fixed functional roc \
  --cp-alphas 0.2 --fixed-thresholds "0.5" --roc-every 50

# UV_PROJECT_ENVIRONMENT=.venv-safe uv run python train.py \
#   --data-path ../data/libero \
#   --task-ids 0 1 2 6 7 9 \
#   --model lstm \
#   --n-epochs 500 \
#   --lr 3e-4 \
#   --seen-train-ratio 0.75 \
#   --unseen-task-ratio 0.33 \
#   --max-per-class 20 \
#   --out-dir logs/safe_lstm \
#   --device cuda \
#   --eval-methods fixed functional roc \
#   --cp-alphas 0.2 --fixed-thresholds "0.5" --roc-every 50
