"""Index the supervised InsPLAD dataset and write resized copies to cache/.

Produces cache/index.csv with columns: path, component, split, folder, cls, group
  split: train / holdout (carved from the original train, grouped by source photo) / test (original val)
"""
import csv
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

from config import CACHE_DIR, CACHE_MAX_SIDE, DATA_DIR, FOLDER_TO_CLASS

HOLDOUT_FRAC = 0.12
SEED = 0


def source_group(name):
    """Identify the drone photo a crop came from, so crops of one photo stay in one split."""
    m = re.match(r"(.*?DJI_\d+)", name)
    return m.group(1) if m else name


def resize_one(args):
    src, dst = args
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.thumbnail((CACHE_MAX_SIDE, CACHE_MAX_SIDE), Image.BICUBIC)
    im.save(dst, quality=95)


def main():
    rows = []
    for comp_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for orig_split in ("train", "val"):
            for folder in sorted((comp_dir / orig_split).iterdir()):
                for img in sorted(folder.glob("*.jpg")):
                    rows.append(dict(src=img, component=comp_dir.name, orig_split=orig_split,
                                     folder=folder.name, cls=FOLDER_TO_CLASS[folder.name],
                                     group=f"{comp_dir.name}/{source_group(img.name)}"))

    # Grouped holdout from train, done per (component, class) so every stratum is represented.
    rng = random.Random(SEED)
    strata = defaultdict(set)
    for r in rows:
        if r["orig_split"] == "train":
            strata[(r["component"], r["cls"])].add(r["group"])
    holdout_groups = set()
    for groups in strata.values():
        groups = sorted(groups)
        rng.shuffle(groups)
        holdout_groups.update(groups[:max(1, round(len(groups) * HOLDOUT_FRAC))])

    for r in rows:
        if r["orig_split"] == "val":
            r["split"] = "test"
        else:
            r["split"] = "holdout" if r["group"] in holdout_groups else "train"
        r["path"] = CACHE_DIR / r["component"] / r["orig_split"] / r["folder"] / r["src"].name

    with ProcessPoolExecutor() as ex:
        list(ex.map(resize_one, [(r["src"], r["path"]) for r in rows], chunksize=64))

    with open(CACHE_DIR / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "component", "split", "folder", "cls", "group"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    print(f"{len(rows)} images indexed")
    for split in ("train", "holdout", "test"):
        c = Counter((r["component"], r["cls"]) for r in rows if r["split"] == split)
        print(f"\n[{split}] total={sum(c.values())}")
        for k in sorted(c):
            print(f"  {k[0]:34s} {k[1]:12s} {c[k]}")


if __name__ == "__main__":
    main()
