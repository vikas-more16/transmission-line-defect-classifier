"""Export the trained checkpoint to TorchScript for deployment.

The traced model runs with plain PyTorch (no timm/torchvision), cutting the web app's RAM by ~120 MB,
which keeps it inside small free hosting plans such as Render's 512 MB.

Usage: python export_traced.py [models/best.pt]   ->  models/best_traced.pt
"""
import json
import sys
from pathlib import Path

import torch

from common import ROOT, build_model


def main(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    size = ckpt["img_size"]
    with torch.no_grad():
        traced = torch.jit.trace(model, torch.randn(1, 3, size, size))
    meta = {k: ckpt[k] for k in ("arch", "img_size", "classes", "threshold", "epoch")}
    out = Path(ckpt_path).with_name(Path(ckpt_path).stem + "_traced.pt")
    torch.jit.save(traced, out, _extra_files={"meta.json": json.dumps(meta)})
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "models" / "best.pt")
