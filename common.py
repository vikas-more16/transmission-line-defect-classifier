"""Shared definitions for the InsPLAD fault classifier.

timm and torchvision are imported only where training needs them: together they add ~120 MB of RAM,
which matters on small hosting plans (the web app runs a TorchScript model and needs neither).
"""
import numpy as np
import torch
from PIL import Image

from config import CACHE_DIR, CLASSES, ROOT  # noqa: F401  (re-exported)

IMG_SIZE = 320
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_model(arch, pretrained=True):
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=len(CLASSES))


def eval_transform(size=IMG_SIZE):
    """Resize + to-tensor + normalise. Same result as torchvision's Resize/ToTensor/Normalize on a PIL image
    (both use PIL's bilinear resize), without importing torchvision."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    def tf(img):
        arr = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        return (torch.from_numpy(arr).permute(2, 0, 1) - mean) / std

    return tf


def train_transform(size=IMG_SIZE):
    from torchvision import transforms

    # Mild colour jitter only: rust is largely a colour cue, so no hue shifts.
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_rgb(path):
    return Image.open(path).convert("RGB")


def fault_probability(logits):
    """P(faulty) = 1 - P(good)."""
    return 1.0 - torch.softmax(logits.float(), dim=1)[:, 0]
