"""Train the faulty / not-faulty classifier on the InsPLAD supervised split.

Usage: python train.py [--arch efficientnet_b0] [--epochs 20] [--img-size 320]
"""
import argparse
import csv
import json
import math
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from common import CACHE_DIR, CLASSES, ROOT, build_model, eval_transform, fault_probability, load_rgb, train_transform


class IndexDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(load_rgb(r["path"])), CLASSES.index(r["cls"])


def load_index(split):
    with open(CACHE_DIR / "index.csv") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


@torch.no_grad()
def predict_logits(model, loader, device, tta=True):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x).float()
            if tta:
                logits = (logits + model(torch.flip(x, dims=[3])).float()) / 2
        out.append(logits.cpu())
    return torch.cat(out)


def pick_threshold(y_true, p_fault):
    """Threshold maximising Youden's J (= balanced accuracy) on the holdout set."""
    fpr, tpr, thr = roc_curve(y_true, p_fault)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows, hold_rows = load_index("train"), load_index("holdout")
    train_dl = DataLoader(IndexDataset(train_rows, train_transform(args.img_size)), batch_size=args.batch_size,
                          shuffle=True, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    hold_dl = DataLoader(IndexDataset(hold_rows, eval_transform(args.img_size)), batch_size=64,
                         num_workers=8, pin_memory=True, persistent_workers=True)
    hold_y = np.array([CLASSES.index(r["cls"]) for r in hold_rows])
    hold_fault = (hold_y != 0).astype(int)
    print(f"train={len(train_rows)} holdout={len(hold_rows)} device={device}")

    model = build_model(args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_dl)
    warmup = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    counts = np.bincount([CLASSES.index(r["cls"]) for r in train_rows], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out_path = ROOT / (args.out or f"models/{args.arch}_{args.img_size}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_score, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)

        logits = predict_logits(model, hold_dl, device, tta=False)
        p = fault_probability(logits).numpy()
        auc = roc_auc_score(hold_fault, p)
        bacc = (((p >= 0.5) & (hold_fault == 1)).sum() / hold_fault.sum()
                + ((p < 0.5) & (hold_fault == 0)).sum() / (hold_fault == 0).sum()) / 2
        type_acc = (logits.argmax(1).numpy() == hold_y).mean()
        score = auc + bacc
        history.append(dict(epoch=epoch, loss=total / n, holdout_auc=float(auc), holdout_bacc=float(bacc),
                            holdout_type_acc=float(type_acc)))
        mark = ""
        if score > best_score:
            best_score, mark = score, " *"
            torch.save({"state_dict": model.state_dict(), "arch": args.arch, "img_size": args.img_size,
                        "classes": CLASSES, "epoch": epoch}, out_path)
        print(f"ep {epoch:2d} loss {total / n:.4f} | holdout AUC {auc:.4f} bal-acc {bacc:.4f} "
              f"type-acc {type_acc:.4f} | {time.time() - t0:.0f}s{mark}", flush=True)

    # Calibrate the decision threshold on the holdout set with the best checkpoint (with TTA, as at inference).
    ckpt = torch.load(out_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    p = fault_probability(predict_logits(model, hold_dl, device, tta=True)).numpy()
    ckpt["threshold"] = pick_threshold(hold_fault, p)
    ckpt["history"] = history
    torch.save(ckpt, out_path)
    print(f"best epoch {ckpt['epoch']}, threshold {ckpt['threshold']:.3f}, saved {out_path}")
    with open(out_path.with_suffix(".history.json"), "w") as f:
        json.dump(history, f, indent=1, default=float)


if __name__ == "__main__":
    main()
