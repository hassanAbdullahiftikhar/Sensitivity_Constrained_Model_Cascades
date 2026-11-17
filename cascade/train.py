"""Train one model on one dataset/seed, keep the best-val-accuracy checkpoint, dump val/test logits.

    python -m cascade.train --dataset pathmnist --arch mobilenet_v3_small --seed 0

Outputs to runs/<dataset>_<size>/<arch>/seed<seed>/: model.pt, logits_val.npz, logits_test.npz, train_log.json
Augmentation runs on the GPU (flips + 90-degree rotations, plus mild colour jitter for dermoscopy).
No label smoothing or mixup: both distort confidence, which is what the routing signals depend on.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .models import IMAGENET_MEAN, IMAGENET_STD, build_model

RUNS = Path(__file__).resolve().parent.parent / "runs"


def run_dir(dataset: str, size: int, arch: str, seed: int) -> Path:
    return RUNS / f"{dataset}_{size}" / arch / f"seed{seed}"


class GpuBatches:
    """Yields normalised float batches from a uint8 copy of the split held in GPU memory
    (PathMNIST-128 is ~5.3 GB as uint8; CPU-side indexing left the GPU ~65% idle)."""

    def __init__(self, split: data.Split, batch_size: int, device: torch.device, train: bool, color_jitter: bool,
                 rng: np.random.Generator):
        self.x = torch.from_numpy(split.images).to(device)
        self.y = torch.from_numpy(split.labels).to(device)
        self.bs, self.dev, self.train, self.jitter, self.rng = batch_size, device, train, color_jitter, rng
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return math.ceil(len(self.y) / self.bs)

    def __iter__(self):
        n = len(self.y)
        order = torch.from_numpy(self.rng.permutation(n) if self.train else np.arange(n)).to(self.dev)
        for i in range(0, n, self.bs):
            idx = order[i:i + self.bs]
            x = self.x[idx].permute(0, 3, 1, 2).float().div_(255)
            if self.train:
                x = self._augment(x)
            yield (x - self.mean) / self.std, self.y[idx]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        flip_h = torch.rand(n, device=x.device) < 0.5
        flip_v = torch.rand(n, device=x.device) < 0.5
        x = torch.where(flip_h.view(-1, 1, 1, 1), x.flip(3), x)
        x = torch.where(flip_v.view(-1, 1, 1, 1), x.flip(2), x)
        k = torch.randint(0, 4, (n,), device=x.device)
        for r in (1, 2, 3):
            m = k == r
            if m.any():
                x[m] = torch.rot90(x[m], r, dims=(2, 3))
        if self.jitter:
            b = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
            c = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
            mu = x.mean(dim=(1, 2, 3), keepdim=True)
            x = ((x - mu) * c + mu) * b
            x = x.clamp_(0, 1)
        return x


@torch.no_grad()
def predict_logits(model, batches: GpuBatches) -> np.ndarray:
    model.eval()
    out = [model(x.contiguous(memory_format=torch.channels_last)).float().cpu() for x, _ in batches]
    return torch.cat(out).numpy()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(data.SPECS))
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=None, help="default: 15 (pathmnist) / 40 (ham10000)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    args = ap.parse_args(argv)

    spec = data.SPECS[args.dataset]
    epochs = args.epochs or (15 if args.dataset == "pathmnist" else 40)
    out = run_dir(args.dataset, args.size, args.arch, args.seed)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = data.load(args.dataset, args.size)
    jitter = args.dataset.startswith("ham10000")
    # "*bal" datasets: inverse-frequency class weights in the loss, checkpoint chosen by balanced accuracy
    # (mean per-class recall) instead of accuracy, which otherwise favours the majority class (nv, ~67%).
    balanced = args.dataset.endswith("bal")
    counts = np.bincount(splits["train"].labels, minlength=spec.num_classes)
    class_w = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=dev) if balanced else None
    val_labels = splits["val"].labels
    train_b = GpuBatches(splits["train"], args.batch_size, dev, True, jitter, rng)
    val_b = GpuBatches(splits["val"], 512, dev, False, False, rng)
    test_b = GpuBatches(splits["test"], 512, dev, False, False, rng)
    for s in splits.values():
        s.images = None  # the GPU copies are all that is needed; frees ~5 GB of host RAM

    model = build_model(args.arch, spec.num_classes).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total, warmup = epochs * len(train_b), len(train_b)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1)))
    )

    log, best_acc = [], -1.0
    for epoch in range(epochs):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in train_b:
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(x.contiguous(memory_format=torch.channels_last)), y, weight=class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
        val_logits = predict_logits(model, val_b)
        pred = val_logits.argmax(1)
        val_acc = float((pred == val_labels).mean())
        val_bacc = float(np.mean([(pred[val_labels == c] == c).mean() for c in np.unique(val_labels)]))
        log.append({"epoch": epoch, "train_loss": loss_sum / n, "val_acc": val_acc, "val_bacc": val_bacc,
                    "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        score = val_bacc if balanced else val_acc
        if score > best_acc:
            best_acc = score
            torch.save(model.state_dict(), out / "model.pt")

    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    for name, b in (("val", val_b), ("test", test_b)):
        np.savez(out / f"logits_{name}.npz", logits=predict_logits(model, b), labels=splits[name].labels)
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "epochs": log, "best_val_acc": best_acc}, indent=1))
    print(f"done: best val acc {best_acc:.4f} -> {out}")


if __name__ == "__main__":
    main()
