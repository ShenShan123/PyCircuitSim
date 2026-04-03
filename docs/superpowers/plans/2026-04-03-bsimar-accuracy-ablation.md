# BSIM-AR Accuracy Ablation Study

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve BSIM-AR Transformer accuracy through 5 sequential improvements, measuring each contribution via test-set NRMSE, with a final pycircuitsim inverter VTC check.

**Architecture:** Each experiment adds one improvement on top of the previous. All experiments train NMOS universal on GPU with the full-size architecture (d=256, 6 layers), 500 epochs, patience=50. The output reordering experiment also requires a column-permutation layer in the data pipeline and metrics.

**Tech Stack:** PyTorch, CUDA, existing nn_model normalization pipeline, DirectLoss.

**Baseline (Exp 0):** d=128/3-layer model, 169 epochs → id NRMSE 17.3%, gds overflow, avg charges ~2%.

---

## File Structure

| File | Responsibility | Tasks |
|------|---------------|-------|
| `external_compact_models/BSIMAR/script/main.py` | CLI entry point, experiment orchestration | 1,2,3,4,5,6 |
| `external_compact_models/BSIMAR/script/model.py` | TransformerEncoderModel with scheduled sampling | 3 |
| `external_compact_models/BSIMAR/script/train.py` | Epoch functions with scheduled sampling + consistency loss | 3,4 |
| `external_compact_models/BSIMAR/script/config.py` | BSIMARConfig extended with new hyperparams | 2,3,4,5 |
| `nn_model/data/normalize.py` | Output reorder/unreorder helpers | 2 |
| `external_compact_models/BSIMAR/script/metrics.py` | Metrics with output reordering awareness | 2 |
| `tests/test_bsimar_ablation.py` | Ablation runner script — runs all 5 experiments | 6 |

---

### Task 1: Exp 1 — Scale Up Model + Training

Train with the full-size architecture. This is the new baseline for all subsequent experiments.

**Files:**
- Modify: `external_compact_models/BSIMAR/script/main.py` (no code changes, just use different CLI args)

- [ ] **Step 1: Train NMOS with full-size architecture**

```bash
conda run -n pycircuitsim python -u -m external_compact_models.BSIMAR.script.main \
    --device-type nmos --universal --cuda \
    --epochs 500 --patience 50 \
    --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 --dropout 0.2 \
    --batch-size 2048 --lr 8e-4 2>&1 | tee results_exp1_nmos.log
```

Checkpoint saved as `ar_universal_nmos_best.pt` (overwrites previous).

- [ ] **Step 2: Record Exp 1 metrics from log output**

Extract the per-target NRMSE table printed at end of training. Save to `external_compact_models/BSIMAR/results/ablation_exp1.txt`.

- [ ] **Step 3: Backup Exp 1 checkpoint**

```bash
cp external_compact_models/BSIMAR/checkpoints/ar_universal_nmos_best.pt \
   external_compact_models/BSIMAR/checkpoints/exp1_nmos_best.pt
cp external_compact_models/BSIMAR/checkpoints/ar_universal_nmos_norm.npz \
   external_compact_models/BSIMAR/checkpoints/exp1_nmos_norm.npz
cp external_compact_models/BSIMAR/checkpoints/ar_universal_nmos_config.npz \
   external_compact_models/BSIMAR/checkpoints/exp1_nmos_config.npz
```

- [ ] **Step 4: Commit**

```bash
git add external_compact_models/BSIMAR/results/ablation_exp1.txt
git commit -m "exp: BSIM-AR Exp 1 — full-size model baseline (d=256, 6 layers)"
```

---

### Task 2: Exp 2 — Reorder Outputs

Put easy-to-predict outputs first (charges → caps → conductances → current) so the autoregressive model has maximum context for the hardest targets.

**Files:**
- Modify: `nn_model/data/normalize.py` — add `BSIMAR_COLUMN_ORDER`, `reorder_outputs()`, `unreorder_outputs()`
- Modify: `external_compact_models/BSIMAR/script/main.py` — add `--reorder` flag, apply reorder before training and unreorder before metrics
- Modify: `external_compact_models/BSIMAR/script/metrics.py` — accept optional column order

- [ ] **Step 1: Add reorder helpers to normalize.py**

Add after `OUTPUT_COLUMN_ORDER` (line 31) in `nn_model/data/normalize.py`:

```python
# BSIM-AR autoregressive order: easy targets first, hardest (id) last.
# Charges (well-behaved, smooth) → capacitances → conductances → current.
BSIMAR_COLUMN_ORDER = [
    "qg", "qd", "qs", "qb",
    "cgg", "cgd", "cgs", "cdg", "cdd",
    "gm", "gds", "gmb",
    "id",
]

# Permutation indices: BSIMAR_COLUMN_ORDER[i] == OUTPUT_COLUMN_ORDER[_REORDER_IDX[i]]
_REORDER_IDX = [OUTPUT_COLUMN_ORDER.index(c) for c in BSIMAR_COLUMN_ORDER]
_UNREORDER_IDX = [BSIMAR_COLUMN_ORDER.index(c) for c in OUTPUT_COLUMN_ORDER]


def reorder_outputs(arr: np.ndarray) -> np.ndarray:
    """Permute columns from OUTPUT_COLUMN_ORDER → BSIMAR_COLUMN_ORDER."""
    return arr[:, _REORDER_IDX]


def unreorder_outputs(arr: np.ndarray) -> np.ndarray:
    """Permute columns from BSIMAR_COLUMN_ORDER → OUTPUT_COLUMN_ORDER."""
    return arr[:, _UNREORDER_IDX]
```

- [ ] **Step 2: Add `--reorder` flag to main.py**

In `main.py` argparse section (after line 86), add:

```python
parser.add_argument("--reorder", action="store_true",
                    help="Reorder outputs for autoregressive (charges→caps→cond→id)")
```

After data loading (after line 119), add reordering logic:

```python
# -- Optional output reordering for autoregressive --
# The model sees columns in BSIMAR_COLUMN_ORDER (charges→caps→cond→id).
# DirectLoss and normalizer always work in OUTPUT_COLUMN_ORDER (original).
# So: reorder dataset → model trains in new order → unreorder pred/targets
# in train/validate loops before passing to DirectLoss.
_reorder_active = False
if args.reorder:
    from nn_model.data.normalize import reorder_outputs, unreorder_outputs
    _reorder_active = True
    # Reorder output columns in all datasets
    train_ds.outputs = torch.tensor(
        reorder_outputs(train_ds.outputs.numpy()), dtype=torch.float32)
    val_ds.outputs = torch.tensor(
        reorder_outputs(val_ds.outputs.numpy()), dtype=torch.float32)
    test_ds.outputs = torch.tensor(
        reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
    print(f"Output columns reordered: charges→caps→cond→id")
```

- [ ] **Step 2b: Add unreorder_fn to train/validate calls**

The critical fix: `DirectLoss` uses hardcoded column indices (col 0=id, cols 4-7=charges, etc.). When `--reorder` is active, the model's predictions and dataset targets are in `BSIMAR_COLUMN_ORDER`. We must unreorder them back to `OUTPUT_COLUMN_ORDER` before passing to `DirectLoss`.

Add an `unreorder_fn` parameter to `train_epoch_direct` and `validate_epoch_direct` in `train.py`:

```python
def train_epoch_direct(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    unreorder_fn=None,  # NEW: callable to unreorder pred/targets before loss
) -> Dict[str, float]:
    """Train one epoch with DirectLoss + teacher forcing."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch, y_batch)  # teacher forcing

        # Unreorder pred and targets for DirectLoss (expects original column order)
        pred_loss = unreorder_fn(pred) if unreorder_fn else pred
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}
```

Same pattern for `validate_epoch_direct` — add `unreorder_fn=None` parameter, apply before `criterion()`.

In the main.py training loop, also pass `unreorder_fn` to the validation function:

```python
        v_losses = val_fn(model, val_loader, criterion, device, unreorder_fn=unreorder_fn)
```

Note: `validate_epoch_direct` signature changes to:
```python
def validate_epoch_direct(model, loader, criterion, device, unreorder_fn=None):
```

In `main.py`, create the unreorder callable and pass it:

```python
# Create unreorder function for loss computation
import torch as _torch
if _reorder_active:
    from nn_model.data.normalize import _UNREORDER_IDX
    _unreorder_idx_t = _torch.tensor(_UNREORDER_IDX, dtype=_torch.long)
    def _unreorder_tensor(t: _torch.Tensor) -> _torch.Tensor:
        return t[:, _unreorder_idx_t.to(t.device)]
    unreorder_fn = _unreorder_tensor
else:
    unreorder_fn = None
```

Pass `unreorder_fn=unreorder_fn` to all train/validate function calls.

Before metrics computation (after `pred_norm, true_norm = test_model(...)`), unreorder back:

```python
if _reorder_active:
    pred_norm = unreorder_outputs(pred_norm)
    true_norm = unreorder_outputs(true_norm)
```

The normalizer stats stay in original order; we reorder only the dataset tensors and unreorder predictions before denorm and before loss.

- [ ] **Step 3: Verify reorder roundtrip**

```bash
conda run -n pycircuitsim python -c "
import numpy as np
from nn_model.data.normalize import (
    OUTPUT_COLUMN_ORDER, BSIMAR_COLUMN_ORDER,
    reorder_outputs, unreorder_outputs,
)
print('Standard:', OUTPUT_COLUMN_ORDER)
print('BSIMAR:  ', BSIMAR_COLUMN_ORDER)
x = np.arange(13).reshape(1, 13).astype(float)
r = reorder_outputs(x)
u = unreorder_outputs(r)
assert np.allclose(x, u), f'Roundtrip failed: {x} != {u}'
# Check that id is last in reordered
assert BSIMAR_COLUMN_ORDER[-1] == 'id'
print('Roundtrip OK, id is position', BSIMAR_COLUMN_ORDER.index('id'))
"
```

Expected: `Roundtrip OK, id is position 12`

- [ ] **Step 4: Train Exp 2**

```bash
conda run -n pycircuitsim python -u -m external_compact_models.BSIMAR.script.main \
    --device-type nmos --universal --cuda \
    --epochs 500 --patience 50 \
    --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 --dropout 0.2 \
    --batch-size 2048 --lr 8e-4 --reorder 2>&1 | tee results_exp2_nmos.log
```

- [ ] **Step 5: Record metrics, backup checkpoint, commit**

```bash
cp external_compact_models/BSIMAR/checkpoints/ar_universal_nmos_best.pt \
   external_compact_models/BSIMAR/checkpoints/exp2_nmos_best.pt
# (same for norm and config)
git add nn_model/data/normalize.py external_compact_models/BSIMAR/script/main.py
git commit -m "exp: BSIM-AR Exp 2 — output reordering (charges→caps→cond→id)"
```

---

### Task 3: Exp 3 — Scheduled Sampling

During training, gradually replace teacher-forced ground-truth tokens with the model's own predictions. This closes the train-test gap.

**Files:**
- Modify: `external_compact_models/BSIMAR/script/model.py` — add `forward_scheduled()` method
- Modify: `external_compact_models/BSIMAR/script/train.py` — add `train_epoch_scheduled()` function
- Modify: `external_compact_models/BSIMAR/script/main.py` — add `--scheduled-sampling` flag, wire in new train function
- Modify: `external_compact_models/BSIMAR/script/config.py` — add `ss_warmup_epochs`, `ss_max_ratio`

- [ ] **Step 1: Add scheduled sampling config**

In `config.py` BSIMARConfig, add after line 57:

```python
    # Scheduled sampling
    ss_warmup_epochs: int = 100   # epochs to ramp from 0 to ss_max_ratio
    ss_max_ratio: float = 0.5     # max fraction of autoregressive tokens
```

- [ ] **Step 2: Add `forward_scheduled()` to model.py**

Add new method to `TransformerEncoderModel` after the `forward()` method (after line 172):

```python
    def forward_scheduled(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ss_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass with scheduled sampling.

        For each target position, with probability ss_ratio, use the model's
        own previous prediction instead of the ground-truth token.
        When ss_ratio=0, this is pure teacher forcing.
        When ss_ratio=1, this is fully autoregressive (but still supervised).

        Args:
            x: (B, input_dim) input features.
            y: (B, target_dim) ground-truth targets.
            ss_ratio: Probability of using model's own prediction at each step.

        Returns:
            (B, target_dim) predicted outputs.
        """
        if ss_ratio <= 0.0:
            return self.forward(x, y)

        batch_size = x.size(0)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        current_seq = torch.cat([x, start_token], dim=1)  # (B, input_dim+1)
        predictions = []

        for t in range(self.target_dim):
            embedded = self.input_projection(current_seq.unsqueeze(-1))
            embedded = self.pos_encoder(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            out = self.transformer_encoder(embedded, mask=causal_mask)
            next_pred = self.output_layer(out[:, -1, :]).squeeze(-1)  # (B,)
            predictions.append(next_pred)

            # Decide next token: model prediction or ground truth
            if t < self.target_dim - 1:
                use_pred = torch.rand(batch_size, device=x.device) < ss_ratio
                next_token = torch.where(
                    use_pred, next_pred.detach(), y[:, t]
                )
                current_seq = torch.cat(
                    [current_seq, next_token.unsqueeze(1)], dim=1
                )

        return torch.stack(predictions, dim=1)  # (B, target_dim)
```

- [ ] **Step 3: Add `train_epoch_scheduled()` to train.py**

Add after `train_epoch_direct()` (after line 80):

```python
def train_epoch_scheduled(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    ss_ratio: float = 0.0,
    unreorder_fn=None,
) -> Dict[str, float]:
    """Train one epoch with scheduled sampling."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model.forward_scheduled(x_batch, y_batch, ss_ratio=ss_ratio)

        # Unreorder for DirectLoss (expects original column order)
        pred_loss = unreorder_fn(pred) if unreorder_fn else pred
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}
```

- [ ] **Step 4: Wire into main.py**

Add `--scheduled-sampling` flag to argparse:

```python
parser.add_argument("--scheduled-sampling", action="store_true",
                    help="Use scheduled sampling (gradual teacher forcing → autoregressive)")
parser.add_argument("--ss-warmup", type=int, default=100,
                    help="Epochs to ramp scheduled sampling ratio (default: 100)")
parser.add_argument("--ss-max-ratio", type=float, default=0.5,
                    help="Max autoregressive ratio (default: 0.5)")
```

Modify the training loop (lines 189-216). Replace the fixed `train_fn` call with:

```python
    for epoch in range(1, args.epochs + 1):
        if args.scheduled_sampling and args.loss == "direct":
            ss_ratio = min(epoch / args.ss_warmup, args.ss_max_ratio)
            from external_compact_models.BSIMAR.script.train import train_epoch_scheduled
            t_losses = train_epoch_scheduled(
                model, train_loader, criterion, optimizer, device,
                ss_ratio=ss_ratio)
        else:
            t_losses = train_fn(model, train_loader, criterion, optimizer, device)
        v_losses = val_fn(model, val_loader, criterion, device)
        # ... rest unchanged
```

- [ ] **Step 5: Train Exp 3**

```bash
conda run -n pycircuitsim python -u -m external_compact_models.BSIMAR.script.main \
    --device-type nmos --universal --cuda \
    --epochs 500 --patience 50 \
    --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 --dropout 0.2 \
    --batch-size 2048 --lr 8e-4 --reorder --scheduled-sampling \
    --ss-warmup 100 --ss-max-ratio 0.5 2>&1 | tee results_exp3_nmos.log
```

- [ ] **Step 6: Record metrics, backup, commit**

```bash
cp external_compact_models/BSIMAR/checkpoints/ar_universal_nmos_best.pt \
   external_compact_models/BSIMAR/checkpoints/exp3_nmos_best.pt
git add external_compact_models/BSIMAR/script/model.py \
        external_compact_models/BSIMAR/script/train.py \
        external_compact_models/BSIMAR/script/main.py \
        external_compact_models/BSIMAR/script/config.py
git commit -m "exp: BSIM-AR Exp 3 — scheduled sampling (warmup=100, max=0.5)"
```

---

### Task 4: Exp 4 — Hybrid Consistency Loss

Add a consistency term that penalizes the gap between teacher-forced and autoregressive predictions on the same batch.

**Files:**
- Modify: `external_compact_models/BSIMAR/script/train.py` — add `train_epoch_hybrid()` function
- Modify: `external_compact_models/BSIMAR/script/main.py` — add `--consistency-weight` flag
- Modify: `external_compact_models/BSIMAR/script/config.py` — add `consistency_weight`

- [ ] **Step 1: Add config param**

In `config.py` BSIMARConfig, add:

```python
    # Hybrid consistency loss
    consistency_weight: float = 0.1  # weight of consistency term
```

- [ ] **Step 2: Add `train_epoch_hybrid()` to train.py**

Add after `train_epoch_scheduled()`. This function combines scheduled sampling with a consistency loss term that penalizes the gap between teacher-forced and autoregressive predictions:

```python
def train_epoch_hybrid(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    ss_ratio: float = 0.0,
    consistency_weight: float = 0.1,
    unreorder_fn=None,
) -> Dict[str, float]:
    """Train with scheduled sampling + consistency loss."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Scheduled-sampling prediction (main supervised loss)
        pred_ss = model.forward_scheduled(x_batch, y_batch, ss_ratio=ss_ratio)

        # Unreorder for DirectLoss (expects original column order)
        pred_loss = unreorder_fn(pred_ss) if unreorder_fn else pred_ss
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)

        # Consistency: compare pure teacher-forcing vs pure autoregressive
        pred_tf = model(x_batch, y_batch)
        with torch.no_grad():
            pred_ar = model(x_batch)
        loss_consistency = torch.nn.functional.mse_loss(pred_tf, pred_ar)

        total_loss = losses["total"] + consistency_weight * loss_consistency
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        total_losses["consist"] = total_losses.get("consist", 0.0) + loss_consistency.item()
        # Track the actual total used for backprop
        total_losses["total_combined"] = total_losses.get("total_combined", 0.0) + total_loss.item()
        n_batches += 1

    avg = {k: v / n_batches for k, v in total_losses.items()}
    avg["total"] = avg.pop("total_combined", avg["total"])
    return avg
```

- [ ] **Step 3: Wire into main.py**

Add `--consistency-weight` flag:

```python
parser.add_argument("--consistency-weight", type=float, default=0.0,
                    help="Weight for TF-vs-AR consistency loss (0=off, default: 0)")
```

In the training loop, replace the train call:

```python
    for epoch in range(1, args.epochs + 1):
        ss_ratio = 0.0
        if args.scheduled_sampling:
            ss_ratio = min(epoch / args.ss_warmup, args.ss_max_ratio)

        if args.consistency_weight > 0 and args.loss == "direct":
            from external_compact_models.BSIMAR.script.train import train_epoch_hybrid
            t_losses = train_epoch_hybrid(
                model, train_loader, criterion, optimizer, device,
                ss_ratio=ss_ratio,
                consistency_weight=args.consistency_weight)
        elif args.scheduled_sampling and args.loss == "direct":
            from external_compact_models.BSIMAR.script.train import train_epoch_scheduled
            t_losses = train_epoch_scheduled(
                model, train_loader, criterion, optimizer, device,
                ss_ratio=ss_ratio)
        else:
            t_losses = train_fn(model, train_loader, criterion, optimizer, device)
```

- [ ] **Step 4: Train Exp 4**

```bash
conda run -n pycircuitsim python -u -m external_compact_models.BSIMAR.script.main \
    --device-type nmos --universal --cuda \
    --epochs 500 --patience 50 \
    --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 --dropout 0.2 \
    --batch-size 2048 --lr 8e-4 \
    --reorder --scheduled-sampling --ss-warmup 100 --ss-max-ratio 0.5 \
    --consistency-weight 0.1 2>&1 | tee results_exp4_nmos.log
```

- [ ] **Step 5: Record metrics, backup, commit**

```bash
cp external_compact_models/BSIMAR/checkpoints/ar_universal_nmos_best.pt \
   external_compact_models/BSIMAR/checkpoints/exp4_nmos_best.pt
git add external_compact_models/BSIMAR/script/train.py \
        external_compact_models/BSIMAR/script/main.py \
        external_compact_models/BSIMAR/script/config.py
git commit -m "exp: BSIM-AR Exp 4 — hybrid consistency loss (w=0.1)"
```

---

### Task 5: Exp 5 — Curriculum on Output Length

Start training with only the first K output targets, gradually increase K to 13 over training. This gives the model a solid foundation on early targets before asking it to predict the full sequence.

**Files:**
- Modify: `external_compact_models/BSIMAR/script/model.py` — add `forward_curriculum()` method
- Modify: `external_compact_models/BSIMAR/script/train.py` — add `train_epoch_curriculum()` function
- Modify: `external_compact_models/BSIMAR/script/main.py` — add `--curriculum` flag
- Modify: `external_compact_models/BSIMAR/script/config.py` — add `curriculum_warmup`

- [ ] **Step 1: Add config param**

In `config.py` BSIMARConfig:

```python
    # Curriculum on output length
    curriculum_warmup: int = 50  # epochs to ramp from 1 target to target_dim
```

- [ ] **Step 2: Add `forward_curriculum()` to model.py**

After `forward_scheduled()`:

```python
    def forward_curriculum(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_targets: int = -1,
        ss_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass predicting only the first n_targets outputs.

        Uses scheduled sampling for the active targets. Inactive targets
        (positions >= n_targets) are filled with teacher-forced ground truth
        to maintain consistent sequence length for positional encoding.

        Args:
            x: (B, input_dim) input features.
            y: (B, target_dim) ground-truth targets.
            n_targets: Number of targets to actively predict (default: all).
            ss_ratio: Scheduled sampling ratio for active positions.

        Returns:
            (B, target_dim) predictions. Positions >= n_targets are copies of y.
        """
        if n_targets <= 0 or n_targets >= self.target_dim:
            return self.forward_scheduled(x, y, ss_ratio=ss_ratio)

        batch_size = x.size(0)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        current_seq = torch.cat([x, start_token], dim=1)
        predictions = []

        for t in range(self.target_dim):
            embedded = self.input_projection(current_seq.unsqueeze(-1))
            embedded = self.pos_encoder(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            out = self.transformer_encoder(embedded, mask=causal_mask)
            next_pred = self.output_layer(out[:, -1, :]).squeeze(-1)

            if t < n_targets:
                predictions.append(next_pred)
            else:
                # Beyond curriculum horizon: use ground truth as output
                predictions.append(y[:, t])

            if t < self.target_dim - 1:
                if t < n_targets:
                    # Scheduled sampling for active targets
                    use_pred = torch.rand(batch_size, device=x.device) < ss_ratio
                    next_token = torch.where(use_pred, next_pred.detach(), y[:, t])
                else:
                    # Teacher forcing for inactive targets
                    next_token = y[:, t]
                current_seq = torch.cat(
                    [current_seq, next_token.unsqueeze(1)], dim=1
                )

        return torch.stack(predictions, dim=1)
```

- [ ] **Step 3: Add `train_epoch_curriculum()` to train.py**

After `train_epoch_hybrid()`:

```python
def train_epoch_curriculum(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_targets: int = 13,
    ss_ratio: float = 0.0,
    consistency_weight: float = 0.0,
    unreorder_fn=None,
) -> Dict[str, float]:
    """Train with curriculum on output length + optional scheduled sampling + consistency.

    Only the first n_targets positions contribute to the loss.
    Positions >= n_targets are masked out to avoid diluting the gradient signal.
    """
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0
    target_dim = None

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if target_dim is None:
            target_dim = y_batch.shape[1]

        optimizer.zero_grad()

        pred = model.forward_curriculum(
            x_batch, y_batch, n_targets=n_targets, ss_ratio=ss_ratio)

        # Mask: only compute loss on active (first n_targets) positions.
        # Replace inactive positions in pred with targets (zero loss).
        if n_targets < target_dim:
            pred_masked = pred.clone()
            pred_masked[:, n_targets:] = y_batch[:, n_targets:]
        else:
            pred_masked = pred

        # Unreorder for DirectLoss
        pred_loss = unreorder_fn(pred_masked) if unreorder_fn else pred_masked
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)

        total_loss = losses["total"]

        if consistency_weight > 0:
            pred_tf = model(x_batch, y_batch)
            with torch.no_grad():
                pred_ar = model(x_batch)
            loss_consist = torch.nn.functional.mse_loss(pred_tf, pred_ar)
            total_loss = total_loss + consistency_weight * loss_consist
            total_losses["consist"] = total_losses.get("consist", 0.0) + loss_consist.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}
```

- [ ] **Step 4: Wire into main.py**

Add flag:

```python
parser.add_argument("--curriculum", action="store_true",
                    help="Use curriculum on output length (ramp from 1 to 13 targets)")
parser.add_argument("--curriculum-warmup", type=int, default=50,
                    help="Epochs to ramp curriculum from 1 to all targets (default: 50)")
```

Replace the training loop train call with a unified dispatcher:

```python
    for epoch in range(1, args.epochs + 1):
        ss_ratio = 0.0
        if args.scheduled_sampling:
            ss_ratio = min(epoch / args.ss_warmup, args.ss_max_ratio)

        n_targets = output_dim  # default: all targets
        if args.curriculum:
            n_targets = max(1, int(output_dim * min(epoch / args.curriculum_warmup, 1.0)))

        if args.curriculum or (args.consistency_weight > 0) or args.scheduled_sampling:
            from external_compact_models.BSIMAR.script.train import train_epoch_curriculum
            t_losses = train_epoch_curriculum(
                model, train_loader, criterion, optimizer, device,
                n_targets=n_targets,
                ss_ratio=ss_ratio,
                consistency_weight=args.consistency_weight,
                unreorder_fn=unreorder_fn)
        else:
            t_losses = train_fn(model, train_loader, criterion, optimizer, device,
                                unreorder_fn=unreorder_fn)
```

This unifies Tasks 3, 4, and 5 into a single `train_epoch_curriculum()` call with the appropriate flags. The separate `train_epoch_scheduled()` and `train_epoch_hybrid()` from earlier tasks remain for backward compatibility but the unified path is preferred.

- [ ] **Step 5: Train Exp 5**

```bash
conda run -n pycircuitsim python -u -m external_compact_models.BSIMAR.script.main \
    --device-type nmos --universal --cuda \
    --epochs 500 --patience 50 \
    --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 --dropout 0.2 \
    --batch-size 2048 --lr 8e-4 \
    --reorder --scheduled-sampling --ss-warmup 100 --ss-max-ratio 0.5 \
    --consistency-weight 0.1 \
    --curriculum --curriculum-warmup 50 2>&1 | tee results_exp5_nmos.log
```

- [ ] **Step 6: Record metrics, backup, commit**

```bash
cp external_compact_models/BSIMAR/checkpoints/ar_universal_nmos_best.pt \
   external_compact_models/BSIMAR/checkpoints/exp5_nmos_best.pt
git add external_compact_models/BSIMAR/script/model.py \
        external_compact_models/BSIMAR/script/train.py \
        external_compact_models/BSIMAR/script/main.py \
        external_compact_models/BSIMAR/script/config.py
git commit -m "exp: BSIM-AR Exp 5 — output curriculum (warmup=50 epochs)"
```

---

### Task 6: Ablation Summary + Final pycircuitsim Test

Compare all 5 experiments and run the best model through pycircuitsim.

**Files:**
- Create: `tests/test_bsimar_ablation.py` — summary script
- Modify: `examples/bsimar_inverter_dc.sp` (already exists from prior work)

- [ ] **Step 1: Create ablation summary script**

```python
"""Compare BSIM-AR ablation experiment results."""
import sys
from pathlib import Path

# Experiment labels and their log files
EXPERIMENTS = [
    ("Exp0: small baseline (d=128,3L)", "results_exp0_nmos.log"),
    ("Exp1: full-size (d=256,6L)", "results_exp1_nmos.log"),
    ("Exp2: + output reorder", "results_exp2_nmos.log"),
    ("Exp3: + scheduled sampling", "results_exp3_nmos.log"),
    ("Exp4: + consistency loss", "results_exp4_nmos.log"),
    ("Exp5: + curriculum", "results_exp5_nmos.log"),
]

def parse_nrmse(log_path: str) -> dict:
    """Extract per-target NRMSE from training log."""
    metrics = {}
    with open(log_path) as f:
        lines = f.readlines()
    in_table = False
    for line in lines:
        if "NRMSE%" in line and "MRE%" in line:
            in_table = True
            continue
        if in_table and "|" in line and "---" not in line and "AVG" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                name = parts[0]
                try:
                    nrmse = float(parts[1])
                    metrics[name] = nrmse
                except ValueError:
                    pass
        if in_table and "AVG" in line:
            break
    return metrics

def main():
    print(f"{'Experiment':<35s} | {'id':>8s} | {'gm':>8s} | {'gds':>8s} | {'charges':>8s} | {'caps':>8s}")
    print("-" * 90)
    for label, logfile in EXPERIMENTS:
        if not Path(logfile).exists():
            print(f"{label:<35s} | {'N/A':>8s}")
            continue
        m = parse_nrmse(logfile)
        id_n = m.get("id", float("nan"))
        gm_n = m.get("gm", float("nan"))
        gds_n = m.get("gds", float("nan"))
        charges = sum(m.get(q, 0) for q in ["qg","qd","qs","qb"]) / 4
        caps = sum(m.get(c, 0) for c in ["cgg","cgd","cgs","cdg","cdd"]) / 5
        print(f"{label:<35s} | {id_n:8.2f} | {gm_n:8.2f} | {gds_n:8.2f} | {charges:8.2f} | {caps:8.2f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train PMOS with best configuration**

After identifying the best experiment, train PMOS with the same flags:

```bash
conda run -n pycircuitsim python -u -m external_compact_models.BSIMAR.script.main \
    --device-type pmos --universal --cuda \
    --epochs 500 --patience 50 \
    --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 --dropout 0.2 \
    --batch-size 2048 --lr 8e-4 \
    <best flags from ablation> 2>&1 | tee results_best_pmos.log
```

- [ ] **Step 3: Run pycircuitsim inverter VTC**

Use the existing `examples/bsimar_inverter_dc.sp` (LEVEL=74, ASAP7 RVT, VDD=0.7V):

```bash
conda run -n pycircuitsim python main.py examples/bsimar_inverter_dc.sp 2>&1
```

- [ ] **Step 4: Compare inverter VTC vs BSIM-CMG**

```bash
conda run -n pycircuitsim python -c "
import numpy as np
ar = np.loadtxt('results/bsimar_inverter_dc/dc/bsimar_inverter_dc_dc_sweep.csv', delimiter=',', skiprows=1)
ref = np.loadtxt('results/bsimcmg_inverter_dc_asap7_ref/dc/bsimcmg_inverter_dc_asap7_ref_dc_sweep.csv', delimiter=',', skiprows=1)
vout_ar, vout_ref = ar[:, 2], ref[:min(len(ar),len(ref)), 2]
n = min(len(vout_ar), len(vout_ref))
good = (vout_ar[:n] >= -0.1) & (vout_ar[:n] <= 0.8)
err = np.abs(vout_ar[:n][good] - vout_ref[:n][good])
nrmse = np.sqrt(np.mean(err**2)) / 0.7 * 100
print(f'Inverter VTC NRMSE: {nrmse:.2f}% ({good.sum()}/{n} good points, {n-good.sum()} outliers)')
"
```

Previous result: 5.28% NRMSE, 3 outlier points.

- [ ] **Step 5: Write ablation summary table and commit**

```bash
conda run -n pycircuitsim python tests/test_bsimar_ablation.py > external_compact_models/BSIMAR/results/ablation_summary.txt
git add tests/test_bsimar_ablation.py external_compact_models/BSIMAR/results/ablation_summary.txt
git commit -m "exp: BSIM-AR ablation study complete — 5 experiments + inverter VTC"
```

---

## Experiment Matrix Summary

| Exp | Change | Flags | Builds on |
|-----|--------|-------|-----------|
| 0 | Baseline (already done) | `--d-model 128 --num-layers 3` | — |
| 1 | Scale up | `--d-model 256 --num-layers 6` | — |
| 2 | + Output reorder | + `--reorder` | Exp 1 |
| 3 | + Scheduled sampling | + `--scheduled-sampling` | Exp 2 |
| 4 | + Consistency loss | + `--consistency-weight 0.1` | Exp 3 |
| 5 | + Curriculum | + `--curriculum` | Exp 4 |

Each experiment trains for up to 500 epochs with patience=50 on NMOS universal data. Estimated time per experiment: 1-2 hours on RTX 6000 (autoregressive validation is the bottleneck). Total ablation: ~6-10 hours.
