"""Bundle the web app for Hugging Face Spaces and (optionally) upload it.

  python deploy_space.py                          # just build ./space_build/ to inspect
  python deploy_space.py --push USER/SPACE_NAME   # build and upload (run `hf auth login` first)

The Space runs on a free CPU machine; the model is small enough (16 MB) that each image takes well under a second.
"""
import argparse
import shutil

from config import ROOT

BUILD = ROOT / "space_build"
FILES = ["app.py", "predict.py", "common.py", "config.py", "models/best.pt"]

SPACE_README = """---
title: Transmission Line Defect Classifier
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.12"
app_file: app.py
pinned: false
---

Upload a drone image of a power-line component and the model tells you whether it is **faulty** or **not faulty**,
and which defect it found (rust, missing cap, bird nest).

EfficientNet-B0 trained on the InsPLAD dataset. Test set: ROC-AUC 0.995, catches 96% of faults.
"""

# CPU-only PyTorch wheels are ~10x smaller than the CUDA ones and Spaces' free tier has no GPU.
SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
timm
pillow
"""


def build():
    import gradio

    if BUILD.exists():
        shutil.rmtree(BUILD)
    for f in FILES:
        (BUILD / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, BUILD / f)
    shutil.copytree(ROOT / "examples", BUILD / "examples")
    (BUILD / "README.md").write_text(SPACE_README.format(gradio_version=gradio.__version__))
    (BUILD / "requirements.txt").write_text(SPACE_REQUIREMENTS)
    print(f"built {BUILD}:")
    for p in sorted(BUILD.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(BUILD)}  ({p.stat().st_size / 1e6:.2f} MB)")


def push(repo_id, private):
    from huggingface_hub import HfApi

    api = HfApi()
    url = api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=BUILD, commit_message="Deploy app")
    print(f"\nuploaded. The Space builds in a few minutes; then open:\n  {url}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="USER/SPACE_NAME", help="upload to this Hugging Face Space")
    ap.add_argument("--private", action="store_true", help="make the Space private (only you can open it)")
    args = ap.parse_args()
    build()
    if args.push:
        push(args.push, args.private)
