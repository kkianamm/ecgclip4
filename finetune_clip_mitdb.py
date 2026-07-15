"""Fine-tune the BiomedCLIP vision tower on MIT-BIH beat images.

Unlike ordinary CLIP InfoNCE training, this script does not treat two images
with the same beat label as negatives. It constructs one prompt-ensembled text
prototype per AAMI class and trains images against those class prototypes using
cross-entropy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config_mitdb as C
from model_utils import get_device, load_biomedclip
from prepare_mitdb import image_path_for
from zero_shot_eval_mitdb import build_class_text_features


class MITDBImageDataset(Dataset):
    def __init__(self, dataframe, preprocess):
        self.dataframe = dataframe
        self.preprocess = preprocess
        self.ids = dataframe.index.to_list()
        self.targets = dataframe[C.CLASSES].to_numpy().argmax(axis=1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ecg_id = int(self.ids[index])
        image = Image.open(image_path_for(ecg_id)).convert("RGB")
        return self.preprocess(image), int(self.targets[index])


@torch.no_grad()
def evaluate(model, loader, text_features, device):
    model.eval()
    predictions = []
    targets = []

    for images, labels in loader:
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = model.logit_scale.exp() * image_features @ text_features.T
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())

    return f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=C.FT_EPOCHS)
    parser.add_argument("--lr", type=float, default=C.FT_LR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=C.NUM_WORKERS)
    parser.add_argument(
        "--output",
        default=str(Path(C.CKPT_DIR) / "biomedclip_mitdb_ft.pt"),
    )
    args = parser.parse_args()

    torch.manual_seed(C.SEED)
    device = get_device()
    print(f"Device: {device}")

    labels = pd.read_csv(
        Path(C.WORK_DIR) / "labels.csv",
        index_col="ecg_id",
    )
    train_df = labels[labels["split"] == "train"]
    val_df = labels[labels["split"] == "val"]
    if args.limit is not None:
        train_df = train_df.head(args.limit)
        val_df = val_df.head(max(1, args.limit // 5))

    model, preprocess, tokenizer = load_biomedclip(device)

    # Compute fixed class text prototypes before switching to train mode.
    text_features = build_class_text_features(
        model,
        tokenizer,
        device,
    ).detach()

    # Tune only the vision tower and CLIP temperature.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.visual.parameters():
        parameter.requires_grad_(True)
    model.logit_scale.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    train_dataset = MITDBImageDataset(train_df, preprocess)
    val_dataset = MITDBImageDataset(val_df, preprocess)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    counts = torch.bincount(
        torch.tensor(train_dataset.targets),
        minlength=len(C.CLASSES),
    ).float()
    class_weights = counts.sum() / counts.clamp_min(1.0)
    class_weights = (class_weights / class_weights.mean()).to(device)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=C.FT_WEIGHT_DECAY,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{args.epochs}",
        )
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = model.logit_scale.exp() * image_features @ text_features.T
                loss = F.cross_entropy(
                    logits,
                    targets,
                    weight=class_weights,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                model.logit_scale.clamp_(0.0, np.log(100.0))

            batch_size = len(images)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_macro_f1 = evaluate(
            model,
            val_loader,
            text_features,
            device,
        )
        mean_loss = running_loss / max(seen, 1)
        print(
            f"epoch {epoch + 1}: mean loss {mean_loss:.4f}, "
            f"val macro F1 {val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No fine-tuned checkpoint was produced.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)
    print(f"Saved best fine-tuned model -> {output_path}")
    print(
        "Evaluate with:\n"
        f"python zero_shot_eval_mitdb.py --ckpt {output_path}"
    )


if __name__ == "__main__":
    main()
