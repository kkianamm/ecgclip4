"""Train a single-label linear probe on frozen MIT-BIH BiomedCLIP features."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import config_mitdb as C


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split_name: str):
    features = np.load(Path(C.FEAT_DIR) / f"X_{split_name}.npy")
    one_hot = np.load(Path(C.FEAT_DIR) / f"y_{split_name}.npy")
    targets = one_hot.argmax(axis=1).astype(np.int64)
    return torch.from_numpy(features), torch.from_numpy(targets)


def compute_metrics(targets: np.ndarray, logits: np.ndarray) -> dict:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(C.CLASSES), dtype=np.int64)[targets]

    per_class_auroc = {}
    for index, class_name in enumerate(C.CLASSES):
        binary_target = one_hot[:, index]
        if binary_target.min() == binary_target.max():
            per_class_auroc[class_name] = float("nan")
        else:
            per_class_auroc[class_name] = float(
                roc_auc_score(binary_target, probabilities[:, index])
            )

    finite = [x for x in per_class_auroc.values() if np.isfinite(x)]

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "macro_auroc_ovr": float(np.mean(finite)) if finite else float("nan"),
        "per_class_auroc": per_class_auroc,
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=np.arange(len(C.CLASSES)),
        ).tolist(),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=np.arange(len(C.CLASSES)),
            target_names=C.CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=C.LP_EPOCHS)
    parser.add_argument("--lr", type=float, default=C.LP_LR)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output",
        default=str(Path(C.RESULTS_DIR) / "linear_probe_mitdb.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    test_x, test_y = load_split("test")

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    feature_dim = train_x.shape[1]
    classifier = nn.Linear(feature_dim, len(C.CLASSES)).to(device)

    counts = torch.bincount(
        train_y,
        minlength=len(C.CLASSES),
    ).float()
    class_weights = counts.sum() / counts.clamp_min(1.0)
    class_weights = class_weights / class_weights.mean()
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=C.LP_WEIGHT_DECAY,
    )

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x_device = val_x.to(device)
    test_x_device = test_x.to(device)

    best_macro_f1 = -1.0
    best_state = None

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    for epoch in range(args.epochs):
        classifier.train()
        permutation = torch.randperm(len(train_x), generator=generator)

        running_loss = 0.0
        for start in range(0, len(train_x), args.batch_size):
            indices = permutation[start : start + args.batch_size].to(device)
            optimizer.zero_grad()
            logits = classifier(train_x[indices])
            loss = loss_function(logits, train_y[indices])
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(indices)

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_x_device).cpu().numpy()
        val_metrics = compute_metrics(val_y.numpy(), val_logits)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in classifier.state_dict().items()
            }

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            mean_loss = running_loss / max(len(train_x), 1)
            print(
                f"epoch {epoch:03d} loss {mean_loss:.4f} "
                f"val macro F1 {val_metrics['macro_f1']:.4f} "
                f"best {best_macro_f1:.4f}"
            )

    if best_state is None:
        raise RuntimeError("No valid linear-probe checkpoint was produced.")

    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_x_device).cpu().numpy()

    metrics = compute_metrics(test_y.numpy(), test_logits)

    print("\n=== MIT-BIH linear probe ===")
    print(f"accuracy          : {metrics['accuracy']:.4f}")
    print(f"balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"macro F1          : {metrics['macro_f1']:.4f}")
    print(f"macro AUROC OVR   : {metrics['macro_auroc_ovr']:.4f}")
    print("classes:", C.CLASSES)
    print(np.asarray(metrics["confusion_matrix"]))

    checkpoint_path = Path(C.CKPT_DIR) / "linear_probe_mitdb.pt"
    torch.save(
        {
            "state_dict": best_state,
            "feature_mean": mean,
            "feature_std": std,
            "classes": C.CLASSES,
            "seed": args.seed,
            "best_validation_macro_f1": best_macro_f1,
        },
        checkpoint_path,
    )
    print(f"Saved linear head -> {checkpoint_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "dataset": "MIT-BIH Arrhythmia Database",
                "method": "linear_probe",
                "classes": C.CLASSES,
                "seed": args.seed,
                "best_validation_macro_f1": best_macro_f1,
                "metrics": metrics,
            },
            indent=2,
            allow_nan=True,
        )
    )
    print(f"Saved metrics -> {output_path}")


if __name__ == "__main__":
    main()
