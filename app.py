"""Web UI for the transmission-line defect classifier.

Run locally:        python app.py
Temporary share:    python app.py --share     (public *.gradio.live link, valid ~1 week while this runs)
Render / Hugging Face: this file is the entry point (see DEPLOY.md)
"""
import argparse
import csv
import os
import tempfile
from pathlib import Path

import gradio as gr

from config import ROOT
from predict import FaultDetector

# Prefer the TorchScript export (much lower RAM on the server); fall back to the training checkpoint.
MODEL_PATH = ROOT / "models" / "best_traced.pt"
detector = FaultDetector(MODEL_PATH if MODEL_PATH.exists() else ROOT / "models" / "best.pt")

DEFECT_NAMES = {"good": "Good", "rust": "Rust / corrosion", "missing-cap": "Missing cap", "bird-nest": "Bird nest"}
EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*.jpg"))

CSS = """
.verdict {border-radius: 14px; padding: 22px 24px; color: #fff; text-align: center;}
.verdict.faulty {background: #c62828;}
.verdict.ok {background: #2e7d32;}
.verdict .title {font-size: 2.1rem; font-weight: 800; letter-spacing: .04em; line-height: 1.1;}
.verdict .sub {font-size: 1.05rem; margin-top: 8px; opacity: .95;}
.verdict.empty {background: transparent; color: inherit; border: 2px dashed #9e9e9e; opacity: .7;}
"""

EMPTY_VERDICT = '<div class="verdict empty"><div class="sub">Upload an image to see the result</div></div>'


def verdict_html(r):
    pct = f"{r['fault_probability'] * 100:.1f}%"
    if r["faulty"]:
        return (f'<div class="verdict faulty"><div class="title">⚠ FAULTY</div>'
                f'<div class="sub">Defect: <b>{DEFECT_NAMES[r["defect_type"]]}</b> · fault probability {pct}</div></div>')
    return (f'<div class="verdict ok"><div class="title">✔ NOT FAULTY</div>'
            f'<div class="sub">Fault probability {pct}</div></div>')


def classify(image):
    if image is None:
        return EMPTY_VERDICT, None
    r = detector.predict(image)
    probs = {DEFECT_NAMES[c]: p for c, p in r["class_probabilities"].items()}
    return verdict_html(r), probs


def classify_batch(files):
    if not files:
        return [], None
    rows = []
    for f in files:
        path = Path(f if isinstance(f, str) else f.name)
        r = detector.predict(path)
        rows.append([path.name, r["label"], DEFECT_NAMES[r["defect_type"]] if r["faulty"] else "—",
                     round(r["fault_probability"], 3)])
    out = Path(tempfile.mkdtemp()) / "fault_predictions.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "verdict", "defect", "fault_probability"])
        w.writerows(rows)
    return rows, str(out)


with gr.Blocks(title="Transmission Line Defect Classifier") as demo:
    gr.Markdown(
        "# ⚡ Transmission Line Defect Classifier\n"
        "Upload a drone image of a power-line component and the model tells you whether it is "
        "**faulty** or **not faulty**, and which defect it found.\n\n"
        "Works best on **cropped images of a single component**: glass insulator, lightning-rod suspension, "
        "polymer-insulator upper shackle, vari-grip or yoke suspension."
    )
    with gr.Tab("Single image"):
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Component image", sources=["upload", "clipboard", "webcam"],
                               height=380)
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    run_btn = gr.Button("Check for faults", variant="primary")
            with gr.Column(scale=1):
                verdict = gr.HTML(EMPTY_VERDICT)
                probs = gr.Label(label="Class probabilities", num_top_classes=4)
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=img, label="Try an example (from the unseen test set)")
        run_btn.click(classify, img, [verdict, probs])
        img.upload(classify, img, [verdict, probs])
        clear_btn.click(lambda: (None, EMPTY_VERDICT, None), None, [img, verdict, probs])

    with gr.Tab("Many images"):
        files = gr.File(file_count="multiple", file_types=["image"], label="Upload several images")
        batch_btn = gr.Button("Check all", variant="primary")
        table = gr.Dataframe(headers=["image", "verdict", "defect", "fault probability"], label="Results",
                             interactive=False)
        csv_out = gr.File(label="Download results (CSV)")
        batch_btn.click(classify_batch, files, [table, csv_out])

    gr.Markdown(
        f"<small>Model: EfficientNet-B0 trained on the InsPLAD dataset · test ROC-AUC 0.995 · "
        f"catches 96% of faults · decision threshold {detector.threshold:.3f} · "
        f"running on {detector.device.type.upper()}</small>"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = ap.parse_args()
    # On a host like Render, PORT is set and the app must listen on all interfaces.
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    demo.launch(share=args.share, server_name=host, server_port=args.port, theme=gr.themes.Soft(), css=CSS)
