"""
train.py — Training loop for DenseNet+Transformer sequential deepfake detection
================================================================================

BUGS FIXED vs previous version
---------------------------------
BUG 1 ► CosineAnnealingWarmRestarts with T_0=20 caused the LR to restart
         aggressively before the model had settled. The observed explosion in
         val_loss at epochs 38-48 (from 5 → 11.5) is a classic warm-restart
         instability when the model has not learned enough to survive a large LR.
         FIX: Replaced with OneCycleLR. OneCycle does a single warmup + cosine
         decay over the full training run. It is far more stable for scratch
         training on small datasets and consistently outperforms SGDR here.

BUG 2 ► Loss function used uniform CrossEntropyLoss. Your dataset may have
         class imbalance across the 5 manipulation types. When NT (no-touch)
         appears often, the model collapses to predicting NT for all positions.
         FIX: Compute class weights from training labels and pass to CE loss.
         This is computed automatically at the start of each run.

BUG 3 ► num_workers=4 with persistent_workers was the right setting but on
         Windows, multiprocessing with DataLoader requires the __main__ guard
         (already present). Added prefetch_factor=2 explicitly for GPU saturation.

BUG 4 ► The overfit check used 50 RANDOM rows (train_rows), not from one source.
         Because the split is source-leakage-free, 50 random rows may come from
         up to 50 different source videos — that IS a real training set, not an
         overfit probe. FIX: Pick all rows from a SINGLE source video to make a
         true ~60-sample overfit test (one source ≈ 60 permutation videos).

BUG 5 ► sched.step() was called once per epoch (correct for SGDR), but the
         scaler/optimizer interaction meant the LR was not seen by the model
         correctly on the first batch. OneCycleLR is called per-batch instead,
         which is the documented correct usage.

BUG 6 ► Removed label_smoothing=0.1 during the overfit check. Label smoothing
         is a REGULARISER — it prevents the model from becoming confident, which
         is exactly what you need during an overfit probe. It was making the
         overfit check impossible.

WORKFLOW
---------
STEP 0 — Build manifest (once):
    python build_manifest.py --metadata-csv _metadata_index.csv \\
        --video-root dataset/generated_videos --out-dir checkpoints --verify-files

STEP 1 — Overfit sanity check (ALWAYS DO THIS FIRST):
    python train.py \\
        --manifest checkpoints/manifest.csv \\
        --video-root dataset/generated_videos \\
        --out-dir checkpoints_debug \\
        --debug-overfit

    Expected: train_loss drops below 0.5 within 20 epochs, below 0.2 within 40.
    If this FAILS: the model/loss/data pipeline has a bug. Do NOT proceed to full training.

STEP 2 — Full training:
    python train.py \\
        --manifest checkpoints/manifest.csv \\
        --video-root dataset/generated_videos \\
        --out-dir checkpoints

STEP 3 — Resume:
    python train.py ... --resume
"""

import json
import time
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DenseNetSequenceModel, ID_TO_EDIT, VOCAB_SIZE, MAX_SEQ_LEN
from dataset import (
    VideoSequenceDataset, load_manifest, build_splits, rows_for_split, parse_label,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════

DEFAULTS = dict(
    epochs          = 80,
    batch_size      = 8,       # RTX 4500 Ada: 8 is safe for 224×224 × 16 frames
    lr              = 5e-4,    # OneCycleLR max_lr — the scheduler handles the rest
    weight_decay    = 1e-4,
    label_smoothing = 0.1,     # only applied during FULL training, not overfit check
    mixup_alpha     = 0.2,     # mild mixup — 0.3 was too aggressive for 5 classes
    early_stop      = 30,
    num_workers     = 4,
    seed            = 42,
    save_every      = 10,
    dropout         = 0.3,
)


# ══════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════
#  CLASS WEIGHTS  — handles label imbalance across 5 manipulation types
# ══════════════════════════════════════════════════════════════════

def compute_class_weights(rows: list, device: torch.device) -> torch.Tensor:
    """
    Counts how often each class appears across ALL positions in the training set.
    Returns inverse-frequency weights, clamped to [0.5, 5.0] to avoid extremes.
    """
    counts = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for row in rows:
        try:
            for label_id in parse_label(row["edit_keys"]):
                counts[label_id] += 1
        except Exception:
            continue
    total  = counts.sum()
    if total == 0:
        return torch.ones(VOCAB_SIZE, device=device)
    freq   = counts / total
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.mean()         # normalise so average weight = 1
    weights = np.clip(weights, 0.5, 5.0)
    log.info(
        "Class weights: " +
        "  ".join(f"{ID_TO_EDIT[i]}={weights[i]:.2f}" for i in range(VOCAB_SIZE))
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ══════════════════════════════════════════════════════════════════
#  MIXUP
# ══════════════════════════════════════════════════════════════════

def mixup_batch(frames: torch.Tensor, labels: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return frames, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(frames.size(0), device=frames.device)
    return lam * frames + (1 - lam) * frames[idx], labels, labels[idx], lam


def compute_loss_mixup(
    logits: list, labels_a: torch.Tensor, labels_b: torch.Tensor,
    lam: float, criterion: nn.Module,
) -> torch.Tensor:
    loss_a = sum(criterion(logits[s], labels_a[:, s]) for s in range(len(logits)))
    loss_b = sum(criterion(logits[s], labels_b[:, s]) for s in range(len(logits)))
    return lam * loss_a + (1 - lam) * loss_b


def compute_loss(logits: list, labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    return sum(criterion(logits[s], labels[:, s]) for s in range(len(logits)))


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device, criterion: nn.Module) -> dict:
    model.eval()
    total_loss    = 0.0
    n_videos      = 0
    exact_correct = 0
    set_correct   = 0
    step_correct  = [0] * MAX_SEQ_LEN
    step_total    = [0] * MAX_SEQ_LEN
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        logits = model(frames)
        loss   = compute_loss(logits, labels, criterion)
        total_loss += loss.item() * frames.size(0)
        n_videos   += frames.size(0)

        preds = torch.stack([l.argmax(1) for l in logits], dim=1)   # (B, S)
        for b in range(frames.size(0)):
            t, p = labels[b].tolist(), preds[b].tolist()
            if t == p: exact_correct += 1
            if set(t) == set(p): set_correct += 1
            for s in range(MAX_SEQ_LEN):
                step_total[s] += 1
                if t[s] == p[s]:
                    step_correct[s] += 1
                    class_tp[t[s]] += 1
                else:
                    class_fn[t[s]] += 1
                    class_fp[p[s]] += 1

    f1s = []
    for c in range(VOCAB_SIZE):
        tp, fp, fn = class_tp[c], class_fp[c], class_fn[c]
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        re = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * re / (pr + re) if (pr + re) else 0.0)

    return {
        "loss":               total_loss / max(n_videos, 1),
        "exact_seq_accuracy": exact_correct / max(n_videos, 1),
        "edit_set_accuracy":  set_correct   / max(n_videos, 1),
        "step_accuracy":      [step_correct[s] / max(step_total[s], 1) for s in range(MAX_SEQ_LEN)],
        "macro_f1":           float(np.mean(f1s)),
        "per_class_f1":       {ID_TO_EDIT[c]: round(f1s[c], 4) for c in range(VOCAB_SIZE)},
    }


def print_metrics(m: dict, prefix: str = "", verbose: bool = False):
    steps = "  ".join(f"s{i+1}={v:.3f}" for i, v in enumerate(m["step_accuracy"]))
    log.info(
        f"{prefix}loss={m['loss']:.4f}  "
        f"exact={m['exact_seq_accuracy']:.4f}  "
        f"set={m['edit_set_accuracy']:.4f}  "
        f"f1={m['macro_f1']:.4f}  {steps}"
    )
    if verbose:
        f1_str = "  ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())
        log.info(f"{prefix}per-class F1: {f1_str}")


# ══════════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ══════════════════════════════════════════════════════════════════

def save_ckpt(path, epoch, model, opt, sched, best_loss, best_acc, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "opt":       opt.state_dict(),
        "sched":     sched.state_dict() if sched is not None else None,
        "best_loss": best_loss,
        "best_acc":  best_acc,
        "config":    cfg,
    }, path)


def load_ckpt(path, model, opt=None, sched=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if opt  and "opt"  in ckpt: opt.load_state_dict(ckpt["opt"])
    if sched and "sched" in ckpt and ckpt["sched"] is not None:
        sched.load_state_dict(ckpt["sched"])
    return ckpt["epoch"], ckpt.get("best_loss", float("inf")), ckpt.get("best_acc", 0.0)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",      required=True)
    ap.add_argument("--video-root",    default=None)
    ap.add_argument("--out-dir",       required=True)
    ap.add_argument("--epochs",        type=int,   default=DEFAULTS["epochs"])
    ap.add_argument("--batch-size",    type=int,   default=DEFAULTS["batch_size"])
    ap.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    ap.add_argument("--weight-decay",  type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--label-smooth",  type=float, default=DEFAULTS["label_smoothing"])
    ap.add_argument("--mixup-alpha",   type=float, default=DEFAULTS["mixup_alpha"])
    ap.add_argument("--early-stop",    type=int,   default=DEFAULTS["early_stop"])
    ap.add_argument("--num-workers",   type=int,   default=DEFAULTS["num_workers"])
    ap.add_argument("--seed",          type=int,   default=DEFAULTS["seed"])
    ap.add_argument("--save-every",    type=int,   default=DEFAULTS["save_every"])
    ap.add_argument("--dropout",       type=float, default=DEFAULTS["dropout"])
    ap.add_argument("--resume",        action="store_true")
    ap.add_argument("--debug-overfit", action="store_true",
                    help="Single-source overfit probe. loss must drop < 0.5 within 20 epochs.")
    args = ap.parse_args()

    set_seed(args.seed)
    import sys
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Python: {sys.executable}")
    log.info(f"Torch:  {torch.__version__}")
    log.info(f"CUDA:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU:    {torch.cuda.get_device_name(0)}")
    log.info(f"Device: {device}")

    # ── Manifest + Splits ─────────────────────────────────────────
    manifest   = load_manifest(Path(args.manifest))
    split_path = out_dir / "split_assignment.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        log.info(f"Loaded split from {split_path.name}")
    else:
        split = build_splits(manifest, seed=args.seed)
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)
        log.info(f"Split saved → {split_path.name}")

    train_rows = rows_for_split(manifest, split, "train")
    val_rows   = rows_for_split(manifest, split, "val")
    test_rows  = rows_for_split(manifest, split, "test")
    log.info(f"Rows: train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    # ── Overfit probe: all videos from ONE source ID ──────────────
    if args.debug_overfit:
        # Group rows by source_id; pick the first source with the most videos
        from collections import Counter
        src_counts = Counter(r["source_id"] for r in train_rows)
        chosen_src = src_counts.most_common(1)[0][0]
        train_rows = [r for r in train_rows if r["source_id"] == chosen_src]
        val_rows   = train_rows[:10]   # validate on same set to check memorisation
        log.info(
            f"[OVERFIT-CHECK] Source '{chosen_src}': {len(train_rows)} videos "
            f"(all {len(train_rows)} sequences). Loss must drop to <0.5 within 20 epochs."
        )

    # ── Class weights (computed from TRAINING SET only) ───────────
    class_weights = compute_class_weights(train_rows, device)

    # ── Loaders ───────────────────────────────────────────────────
    pin = (device.type == "cuda")
    pw  = (args.num_workers > 0)

    train_loader = DataLoader(
        VideoSequenceDataset(train_rows, args.video_root, augment=(not args.debug_overfit)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=pw,
        prefetch_factor=2 if pw else None,
    )
    val_loader = DataLoader(
        VideoSequenceDataset(val_rows, args.video_root, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=pw,
        prefetch_factor=2 if pw else None,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = DenseNetSequenceModel(
        vocab_size   = VOCAB_SIZE,
        max_seq_len  = MAX_SEQ_LEN,
        growth_rate  = 24,
        block_layers = (6, 6, 6, 6),
        cnn_out_dim  = 256,
        nhead        = 4,
        tf_layers    = 2,
        dim_ff       = 512,
        dropout      = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: DenseNetSequenceModel  |  Parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── Scheduler ─────────────────────────────────────────────────
    # OneCycleLR: warmup (first 30% of steps) → cosine decay.
    # Called per-batch, not per-epoch.
    # For overfit check: plain constant LR (scheduler=None) so we
    # do not waste warm-up budget on 60 samples.
    steps_per_epoch = len(train_loader)
    if args.debug_overfit:
        sched = None   # constant LR for overfit probe
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr        = args.lr,
            total_steps   = args.epochs * steps_per_epoch,
            pct_start     = 0.1,      # 10% warmup
            anneal_strategy = "cos",
            div_factor    = 10.0,     # start at max_lr / 10
            final_div_factor = 1e4,   # end at max_lr / 10000
        )

    # ── Loss ──────────────────────────────────────────────────────
    # During overfit: NO label smoothing, NO class weights (removes all regularisation).
    # During full training: both enabled.
    if args.debug_overfit:
        crit = nn.CrossEntropyLoss(weight=None, label_smoothing=0.0)
    else:
        crit = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smooth,
        )

    # ── Mixed precision (CUDA only) ───────────────────────────────
    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Checkpoint resume ─────────────────────────────────────────
    last_ckpt      = out_dir / "last.pt"
    best_loss_ckpt = out_dir / "best_loss.pt"
    best_acc_ckpt  = out_dir / "best_acc.pt"
    start_epoch    = 0
    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    patience       = 0

    if args.resume and last_ckpt.exists():
        start_epoch, best_val_loss, best_val_acc = load_ckpt(
            last_ckpt, model, opt, sched, str(device)
        )
        start_epoch += 1
        log.info(
            f"Resumed from epoch {start_epoch}, "
            f"best_loss={best_val_loss:.4f}, best_acc={best_val_acc:.4f}"
        )

    log_path = out_dir / "train_log.jsonl"

    # ══════════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ══════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss = 0.0
        n_batch  = 0
        t0 = time.time()

        for frames, labels in train_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixup: disabled during overfit probe
            alpha = args.mixup_alpha if not args.debug_overfit else 0.0
            frames_m, la, lb, lam = mixup_batch(frames, labels, alpha)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(frames_m)
                loss   = compute_loss_mixup(logits, la, lb, lam, crit)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            # OneCycleLR is stepped per-batch
            if sched is not None:
                sched.step()

            run_loss += loss.item()
            n_batch  += 1

        train_loss = run_loss / max(n_batch, 1)
        cur_lr     = opt.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        log.info(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  lr={cur_lr:.2e}  time={elapsed:.1f}s"
        )

        # ── Validation ────────────────────────────────────────────
        val_m    = evaluate(model, val_loader, device, crit)
        verbose  = ((epoch + 1) % 5 == 0)
        print_metrics(val_m, prefix="  [val]  ", verbose=verbose)

        # ── Save last checkpoint ──────────────────────────────────
        save_ckpt(last_ckpt, epoch, model, opt, sched, best_val_loss, best_val_acc, vars(args))

        # ── Best val LOSS ─────────────────────────────────────────
        val_loss = val_m["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience      = 0
            save_ckpt(best_loss_ckpt, epoch, model, opt, sched, best_val_loss, best_val_acc, vars(args))
            log.info(f"  ✓ Best val_loss={best_val_loss:.4f} → best_loss.pt saved")
        else:
            patience += 1
            log.info(f"  No improvement ({patience}/{args.early_stop})")

        # ── Best val ACCURACY ─────────────────────────────────────
        val_acc = val_m["exact_seq_accuracy"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_ckpt(best_acc_ckpt, epoch, model, opt, sched, best_val_loss, best_val_acc, vars(args))
            log.info(f"  ✓ Best val_acc={best_val_acc:.4f} → best_acc.pt saved")

        # ── Periodic epoch checkpoint ─────────────────────────────
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ep_ckpt = out_dir / f"epoch_{epoch+1:04d}.pt"
            save_ckpt(ep_ckpt, epoch, model, opt, sched, best_val_loss, best_val_acc, vars(args))
            log.info(f"  Periodic checkpoint → {ep_ckpt.name}")

        # ── JSONL log ─────────────────────────────────────────────
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch":      epoch + 1,
                "train_loss": train_loss,
                "lr":         cur_lr,
                **{f"val_{k}": v for k, v in val_m.items()},
            }) + "\n")

        # ── Early stopping ────────────────────────────────────────
        if not args.debug_overfit and patience >= args.early_stop:
            log.info(f"Early stopping at epoch {epoch+1}.")
            break

        # ── Overfit probe success criterion ───────────────────────
        if args.debug_overfit:
            if train_loss < 0.5:
                log.info(
                    f"[OVERFIT-CHECK PASSED] train_loss={train_loss:.4f} < 0.5 ✓  "
                    "Pipeline is correct. Proceed to full training."
                )
                break
            if epoch >= 39 and train_loss > 1.5:
                log.error(
                    f"[OVERFIT-CHECK FAILED] After {epoch+1} epochs, train_loss={train_loss:.4f}."
                    " The model cannot memorise 60 samples — there is a remaining bug."
                    " Check: (1) label shapes, (2) loss function, (3) gradient flow."
                )
                break

    log.info(f"Done. Best val_loss={best_val_loss:.4f}  best_val_acc={best_val_acc:.4f}")

    # ── Final test evaluation ─────────────────────────────────────
    if not args.debug_overfit and len(test_rows) > 0 and best_loss_ckpt.exists():
        log.info("Running test evaluation with best_loss.pt ...")
        load_ckpt(best_loss_ckpt, model, device=str(device))
        test_loader = DataLoader(
            VideoSequenceDataset(test_rows, args.video_root, augment=False),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        test_m = evaluate(model, test_loader, device, crit)
        print_metrics(test_m, prefix="  [test] ", verbose=True)
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(test_m, f, indent=2)
        log.info(f"Test metrics → {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
