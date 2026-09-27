"""Evaluate a trained checkpoint on the held-out test set (the dataset's original `val` folders).

Usage: python evaluate.py models/<checkpoint>.pt
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader

from common import CLASSES, build_model, eval_transform, fault_probability
from train import IndexDataset, load_index, predict_logits


def fault_metrics(y, pred, p):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return dict(n=len(y), n_faulty=int(y.sum()), accuracy=float((pred == y).mean()),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                precision=float(prec), recall=float(rec), f1=float(f1),
                roc_auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
                avg_precision=float(average_precision_score(y, p)) if y.sum() else None,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def main(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    thr = ckpt["threshold"]

    rows = load_index("test")
    dl = DataLoader(IndexDataset(rows, eval_transform(ckpt["img_size"])), batch_size=64, num_workers=8)
    logits = predict_logits(model, dl, device, tta=True)
    p = fault_probability(logits).numpy()
    y_type = np.array([CLASSES.index(r["cls"]) for r in rows])
    y = (y_type != 0).astype(int)
    pred = (p >= thr).astype(int)
    comps = np.array([r["component"] for r in rows])

    report = {"checkpoint": str(ckpt_path), "arch": ckpt["arch"], "img_size": ckpt["img_size"], "threshold": thr,
              "overall": fault_metrics(y, pred, p), "per_component": {}}
    for c in sorted(set(comps)):
        m = comps == c
        report["per_component"][c] = fault_metrics(y[m], pred[m], p[m])
    # Defect-type accuracy among components that are truly faulty and were flagged as faulty
    hit = (y == 1) & (pred == 1)
    type_pred = logits[:, 1:].argmax(1).numpy() + 1
    report["defect_type_accuracy_on_detected_faults"] = float((type_pred[hit] == y_type[hit]).mean())

    o = report["overall"]
    print(f"\n=== TEST SET  ({ckpt['arch']} @ {ckpt['img_size']}px, threshold {thr:.3f}) ===")
    print(f"images {o['n']}  faulty {o['n_faulty']}")
    print(f"ROC-AUC {o['roc_auc']:.4f}  AP {o['avg_precision']:.4f}  balanced-acc {o['balanced_accuracy']:.4f}  "
          f"acc {o['accuracy']:.4f}")
    print(f"precision {o['precision']:.4f}  recall {o['recall']:.4f}  F1 {o['f1']:.4f}   "
          f"TP {o['tp']} FP {o['fp']} TN {o['tn']} FN {o['fn']}")
    print(f"defect-type accuracy on detected faults: {report['defect_type_accuracy_on_detected_faults']:.4f}\n")
    print(f"{'component':34s} {'n':>5s} {'faulty':>6s} {'AUC':>6s} {'bAcc':>6s} {'prec':>6s} {'recall':>6s} {'F1':>6s}  FP  FN")
    for c, m in report["per_component"].items():
        print(f"{c:34s} {m['n']:5d} {m['n_faulty']:6d} {m['roc_auc']:6.3f} {m['balanced_accuracy']:6.3f} "
              f"{m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['fp']:3d} {m['fn']:3d}")

    out = Path(ckpt_path).with_suffix(".test_report.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"\nreport saved to {out}")
    return report


if __name__ == "__main__":
    main(sys.argv[1])
