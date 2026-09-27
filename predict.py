"""Predict whether power-line component images are faulty.

Usage:
  python predict.py path/to/image.jpg [more images or folders ...] [--model models/best.pt] [--csv out.csv]

Input images should be crops of a single component (as in the InsPLAD fault-classification set),
e.g. a glass insulator, lightning-rod suspension, polymer-insulator upper shackle, vari-grip or yoke suspension.
"""
import argparse
import csv
from pathlib import Path

import torch

from common import ROOT, build_model, eval_transform, load_rgb

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaultDetector:
    def __init__(self, model_path=ROOT / "models" / "best.pt", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(model_path, map_location=self.device)
        self.classes, self.threshold = ckpt["classes"], ckpt["threshold"]
        self.model = build_model(ckpt["arch"], pretrained=False).to(self.device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.tf = eval_transform(ckpt["img_size"])

    @torch.no_grad()
    def predict(self, image):
        """image: file path or PIL image. Returns a dict with the verdict and probabilities."""
        img = load_rgb(image) if isinstance(image, (str, Path)) else image.convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        logits = (self.model(x) + self.model(torch.flip(x, dims=[3]))) / 2
        probs = torch.softmax(logits.float(), 1)[0].cpu()
        p_fault = float(1 - probs[0])
        faulty = p_fault >= self.threshold
        return {
            "faulty": faulty,
            "label": "FAULTY" if faulty else "NOT FAULTY",
            "fault_probability": round(p_fault, 4),
            "defect_type": self.classes[int(probs[1:].argmax()) + 1] if faulty else None,
            "class_probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
        }


def collect(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
        else:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("--model", default=ROOT / "models" / "best.pt")
    ap.add_argument("--csv", help="optional path to write results as CSV")
    args = ap.parse_args()

    det = FaultDetector(args.model)
    results = []
    for path in collect(args.inputs):
        r = det.predict(path)
        results.append({"image": str(path), **r})
        extra = f" ({r['defect_type']})" if r["faulty"] else ""
        print(f"{r['label']:10s}{extra:15s} p_fault={r['fault_probability']:.3f}  {path}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "label", "fault_probability", "defect_type"])
            for r in results:
                w.writerow([r["image"], r["label"], r["fault_probability"], r["defect_type"] or ""])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
