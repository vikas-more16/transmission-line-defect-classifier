"""Shared definitions for the InsPLAD fault classifier."""
import timm
import torch
from PIL import Image
from torchvision import transforms

from config import CACHE_DIR, CLASSES, ROOT  # noqa: F401  (re-exported)

IMG_SIZE = 320
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_model(arch, pretrained=True):
    return timm.create_model(arch, pretrained=pretrained, num_classes=len(CLASSES))


def eval_transform(size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform(size=IMG_SIZE):
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
