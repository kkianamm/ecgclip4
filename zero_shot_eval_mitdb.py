"""Zero-shot BiomedCLIP evaluation on MIT-BIH AAMI beat classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

import config_mitdb as C
from model_utils import get_device, load_biomedclip
from prepare_mitdb import image_path_for


@torch.no_grad()
def build_class_text_features(model, tokenizer, device):
    features = []
    for class_name in C.CLASSES:
        description = C.CLASS_DESCRIPTIONS[class_name]
        prompts = [template.format(description) for template in C.PROMPT_TEMPLATES]
        tokens = tokenizer(prompts, context_length=C.CONTEXT_LENGTH).to(device)
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embedding = embeddings.mean(dim=0)
        embedding = embedding / embedding.norm()
        features.append(embedding)
    return torch.stack(features, dim=0)


@torch.no_grad()
def encode_images(
    model,
    preprocess,
    device,
    ecg_ids,
    batch_size=C.BATCH_SIZE,
):
    outputs = []
    batch = []

    for ecg_id in tqdm(ecg_ids, desc="Encoding MIT-BIH beat images"):
        path = image_path_for(int(ecg_id))
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing image {path}. Run prepare_mitdb.py without --no-render."
            )
        batch.append(preprocess(Image.open(path).convert("RGB")))

        if len(batch) == batch_size:
            tensor = torch.stack(batch).to(device)
            embedding = model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            outputs.append(embedding.cpu())
            batch = []

    if batch:
        tensor = torch.stack(batch).to(device)
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        outputs.append(embedding.cpu())

    if not outputs:
        raise RuntimeError("No images were encoded.")
    return torch.cat(outputs, dim=0)


def evaluate(y_true, probabilities):
    y_pred = probabilities.argmax(axis=1)
    one_hot = np.eye(len(C.CLASSES), dtype=np.int64)[y_true]

    per_class_auroc = {}
    for index, class_name in enumerate(C.CLASSES):
        target = one_hot[:, index]
        if target.min() == target.max():
            per_class_auroc[class_name] = float("nan")
        else:
            per_class_auroc[class_name] = float(
                roc_auc_score(target, probabilities[:, index])
            )

    finite_aurocs = [
        value for value in per_class_auroc.values() if np.isfinite(value)
    ]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_auroc_ovr": (
            float(np.mean(finite_aurocs)) if finite_aurocs else float("nan")
        ),
        "per_class_auroc": per_class_auroc,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(len(C.CLASSES)),
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(C.CLASSES)),
            target_names=C.CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument(
        "--output",
        default=str(Path(C.RESULTS_DIR) / "zero_shot_mitdb.json"),
    )
    args = parser.parse_args()

    labels_path = Path(C.WORK_DIR) / "labels.csv"
    labels = pd.read_csv(labels_path, index_col="ecg_id")
    subset = labels[labels["split"] == args.split]
    if args.limit is not None:
        subset = subset.head(args.limit)
    if subset.empty:
        raise RuntimeError(f"No rows found for split={args.split}")

    device = get_device()
    print(f"Device: {device}")
    print(f"{args.split} beats: {len(subset):,}")

    model, preprocess, tokenizer = load_biomedclip(
        device,
        ckpt_path=args.ckpt,
    )
    text_features = build_class_text_features(model, tokenizer, device)
    image_features = encode_images(
        model,
        preprocess,
        device,
        subset.index.tolist(),
        batch_size=args.batch_size,
    )

    logit_scale = model.logit_scale.exp().detach()
    logits = logit_scale * image_features.to(device) @ text_features.T
    probabilities = logits.softmax(dim=-1).cpu().numpy()

    y_true = subset[C.CLASSES].to_numpy().argmax(axis=1)
    metrics = evaluate(y_true, probabilities)

    print("\n=== Zero-shot BiomedCLIP on MIT-BIH beats ===")
    print(f"accuracy          : {metrics['accuracy']:.4f}")
    print(f"balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"macro F1          : {metrics['macro_f1']:.4f}")
    print(f"macro AUROC OVR   : {metrics['macro_auroc_ovr']:.4f}")
    print("\nConfusion matrix, rows=true and columns=predicted")
    print("classes:", C.CLASSES)
    print(np.asarray(metrics["confusion_matrix"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "split": args.split,
        "n_samples": len(subset),
        "classes": C.CLASSES,
        "checkpoint": args.ckpt,
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(f"Saved metrics -> {output_path}")


if __name__ == "__main__":
    main()
