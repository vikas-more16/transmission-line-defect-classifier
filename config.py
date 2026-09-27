"""Paths and label definitions (no torch dependency, so data prep can run anywhere)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "InsPLAD_Dataset" / "supervised_fault_classification" / "defect_supervised"
CACHE_DIR = ROOT / "cache"          # resized copies of the dataset for fast loading
CACHE_MAX_SIDE = 448

# Defect-type classes predicted by the model. Class 0 is "good"; everything else is a fault.
CLASSES = ["good", "rust", "missing-cap", "bird-nest"]
FOLDER_TO_CLASS = {
    "good": "good", "normal": "good",
    "rust": "rust", "corrosão": "rust",
    "missing-cap": "missing-cap",
    "bird-nest": "bird-nest",
}
