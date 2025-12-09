# Recursion

This repo contains experiments with the goal of finding new and better ways to train RNNs, and / or
new recurrent architectures for modern AI problems.

It is based on the [TRM repo](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

Recommendations:
- Using `uv`: https://docs.astral.sh/uv/
- Using `pyenv` to manage python versions: https://github.com/pyenv/pyenv

```bash```
Standard install (no GPU or majority of recent Nvidia GPUs):
```bash
uv venv
uv pip install -e .
```

With Nvidia GTX 1080 GPU (need to force CUDA version to be 12.6):
```bash
uv venv
uv pip install -e . --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu126
```

If you want the logger to sync results to your Weights & Biases (https://wandb.ai/):
```bash
wandb login YOUR-LOGIN
```

### Dataset Preparation

```bash
# ARC-AGI-1
uv run python -m recursion.dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
uv run python -m recursion.dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2 training data contains some ARC-AGI-1 eval data

# Sudoku-Extreme
uv run python -m recursion.dataset.build_sudoku_dataset --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Maze-Hard
uv run python -m recursion.dataset.build_maze_dataset.py # 1000 examples, 8 augments
```

## Experiments

### Sudoku-Extreme (assuming 1 L40S GPU):

```bash
run_name="pretrain_mlp_t_sudoku"
uv run python -m recursion.pretrain \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

Expected: Around 87% exact-accuracy (+- 2%)

```bash
run_name="pretrain_att_sudoku"
uv run python recursion.pretrain \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

Expected: Around 75% exact-accuracy (+- 2%)

*Runtime:* < 20 hours

### Maze-Hard (assuming 4 L40S GPUs):

```bash
run_name="pretrain_att_maze30x30"
uv run torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 src/recursion/pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Runtime:* < 24 hours

Actually, you can run Maze-Hard with 1 L40S GPU by reducing the batch-size with no noticable loss in performance:

```bash
run_name="pretrain_att_maze30x30_1gpu"
uv run python -m pretrain \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Runtime:* < 24 hours


### ARC-AGI-1 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc1concept_4"
uv run torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 src/recursion/pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### ARC-AGI-2 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc2concept_4"
uv run torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 src/recursion/pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days
