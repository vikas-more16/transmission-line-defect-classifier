# transmission-line-defect-classifier

A deep learning model that looks at a drone image of a power-line component and decides whether it is **FAULTY** or **NOT FAULTY**. If it is faulty, the model also names the defect type: **rust**, **missing cap** or **bird nest**.

It is built with PyTorch and a pretrained EfficientNet (via `timm`), and trained on the supervised fault-classification split of the [InsPLAD dataset](https://github.com/andreluizbvs/InsPLAD).

```
$ python predict.py samples/
NOT FAULTY                p_fault=0.194  samples/glass_insulator_good.jpg
FAULTY     (missing-cap)  p_fault=0.998  samples/glass_insulator_missing_cap.jpg
FAULTY     (bird-nest)    p_fault=0.969  samples/vari_grip_bird_nest.jpg
FAULTY     (rust)         p_fault=0.990  samples/yoke_suspension_rust.jpg
NOT FAULTY                p_fault=0.229  samples/lightning_rod_good.jpg
```

## Supported components

The model is trained on cropped images of these five component types:

| Component | Defects it detects |
|---|---|
| Glass insulator | missing cap |
| Lightning-rod suspension | rust |
| Polymer-insulator upper shackle | rust / corrosion |
| Vari-grip | rust, bird nest |
| Yoke suspension | rust |

Input images should show **one component, cropped**, as in the dataset. The model is not built for full drone photos showing many components. For those, first run an object detector (for example, one trained on `InsPLAD-det`) to crop out each component.

## Project structure

```
transmission-line-defect-classifier/
├── config.py          # paths, class names, label mapping
├── common.py          # model builder, image transforms
├── prepare_data.py    # indexes the dataset, makes train/holdout/test splits, caches resized images
├── train.py           # trains the model and picks the decision threshold
├── evaluate.py        # full metrics on the test set
├── predict.py         # run the trained model on your own images
├── app.py             # web UI (Gradio)
├── export_traced.py   # exports the model to TorchScript (low-RAM version used by the web app)
├── render.yaml        # Render deployment settings
├── requirements-render.txt  # packages for the hosted app (CPU-only)
├── deploy_space.py    # bundle + upload the web UI to Hugging Face Spaces (needs PRO)
├── examples/          # sample images shown in the web UI
└── models/            # trained checkpoints (.pt) and test reports (.json)
```

## Setup

Requires Python 3.10 or newer. An NVIDIA GPU is strongly recommended for training; prediction also works on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Dataset

Download InsPLAD and place it next to this folder, so the layout looks like this:

```
<parent>/
├── transmission-line-defect-classifier/   # this repo
└── InsPLAD_Dataset/
    └── supervised_fault_classification/
        └── defect_supervised/
            ├── glass-insulator/{train,val}/...
            ├── lightning-rod-suspension/{train,val}/...
            └── ...
```

To use a different location, change `DATA_DIR` in `config.py`.

## Usage

### 1. Prepare the data

```bash
python prepare_data.py
```

This indexes all 11,525 images, writes resized copies to `cache/` (for much faster training) and creates three splits:

| Split | Source | Images | Used for |
|---|---|---|---|
| train | dataset `train/` (88%) | 4,486 | training |
| holdout | dataset `train/` (12%) | 622 | choosing the best epoch and the decision threshold |
| test | dataset `val/` | 6,417 (146 faulty) | final evaluation only |

The holdout is grouped by source drone photo, so crops from the same photo never appear in both training and holdout.

Folder names are mapped to labels as follows. `good` and `normal` mean **not faulty**. `rust` and `corrosão` (Portuguese for "corrosion") mean **rust**. `missing-cap` and `bird-nest` keep their own labels.

### 2. Train

```bash
python train.py --out models/best.pt
```

Useful options:

| Option | Default | Notes |
|---|---|---|
| `--arch` | `efficientnet_b0` | any `timm` model name, e.g. `convnext_tiny`, `efficientnet_b3` |
| `--epochs` | `20` | |
| `--img-size` | `320` | input resolution |
| `--batch-size` | `32` | lower it if you run out of GPU memory |
| `--lr` | `3e-4` | AdamW with warmup and cosine decay |

After each epoch, the model is scored on the holdout set (ROC-AUC plus balanced accuracy), and the best one is saved. The faulty/not-faulty threshold is then tuned on the holdout set and stored inside the checkpoint.

### 3. Evaluate on the test set

```bash
python evaluate.py models/best.pt
```

This prints overall and per-component metrics and saves them to `models/best.test_report.json`.

### 4. Predict on new images

```bash
# single image
python predict.py path/to/image.jpg

# whole folder, saving results to a CSV
python predict.py path/to/folder/ --csv results.csv

# a different checkpoint
python predict.py image.jpg --model models/other.pt
```

From Python:

```python
from predict import FaultDetector

detector = FaultDetector("models/best.pt")
result = detector.predict("image.jpg")
print(result)
# {'faulty': True, 'label': 'FAULTY', 'fault_probability': 0.973,
#  'defect_type': 'rust', 'class_probabilities': {...}}
```

### 5. Web app

```bash
python app.py            # open http://127.0.0.1:7860
python app.py --share    # temporary public link
```

Upload or paste an image and get **FAULTY / NOT FAULTY** with the defect type. The "Many images" tab checks a batch and exports a CSV. See [DEPLOY.md](DEPLOY.md) for free permanent hosting on Render.

## How the model works

- **Backbone:** EfficientNet-B0 pretrained on ImageNet, fine-tuned on all five component types together.
- **Output:** four classes: `good`, `rust`, `missing-cap` and `bird-nest`. The fault probability is `1 − P(good)`. If that is at or above the tuned threshold, the component is flagged **FAULTY**, and the highest-scoring defect class gives the defect type.
- **Training details:** mixed-precision training, class-weighted cross-entropy with label smoothing, flip/rotation/crop augmentation, and only mild colour jitter (no hue shifts, since rust is largely a colour cue).
- **Inference:** each image is scored twice (original and horizontally flipped) and the results are averaged.

## Results

The model is EfficientNet-B0 at 320 px, trained for 20 epochs. The best checkpoint came from epoch 13, and the decision threshold is 0.527. It was tested on the dataset's `val` folders: **6,417 images the model never saw during training or tuning**, 146 of them faulty.

| Metric | Value |
|---|---|
| ROC-AUC | **0.995** |
| Balanced accuracy | **97.7%** |
| Recall (faults caught) | **95.9%** (140 / 146) |
| Precision (flags that were real faults) | 79.6% |
| F1 (faulty) | 0.870 |
| Accuracy | 99.4% |
| Defect-type accuracy (on caught faults) | 100% |

Confusion matrix: 140 faults caught, 6 faults missed, 36 false alarms, 6,235 correct "not faulty".

Per component:

| Component | Images | Faulty | ROC-AUC | Recall | Precision | Missed | False alarms |
|---|---|---|---|---|---|---|---|
| Glass insulator | 59 | 30 | 0.962 | 83.3% | 96.2% | 5 | 1 |
| Lightning-rod suspension | 251 | 20 | 1.000 | 100% | 100% | 0 | 0 |
| Polymer-insulator upper shackle | 64 | 33 | 0.999 | 97.0% | 100% | 1 | 0 |
| Vari-grip | 281 | 43 | 0.993 | 100% | 78.2% | 0 | 12 |
| Yoke suspension | 5,762 | 20 | 1.000 | 100% | 46.5% | 0 | 23 |

**How to read this:** the test set is very imbalanced. Yoke suspension alone has 5,742 good images and only 20 rusted ones. Most false alarms come from that component, yet they are only 0.4% of its good images (23 of 5,742). Across all components, the model catches 96% of faults.

For inspection work, a missed fault usually costs more than an extra look at a flagged image. If you need fewer false alarms instead, raise the threshold (the `threshold` value stored in the checkpoint).

The full metrics are in `models/best.test_report.json`.

## Limitations

- Only the five component types above were seen in training. Results on other components (dampers, spacers, plates, and so on) are unreliable.
- The model expects cropped single-component images, similar in framing to the dataset.
- Some defect classes have few test examples (as few as 20), so their per-component metrics carry wide uncertainty.

## Acknowledgements

Dataset: **InsPLAD**, *Power Line Assets Inspection Dataset* (Vieira-e-Silva et al., 2023). Please cite the original authors and follow the dataset's license if you use it.
