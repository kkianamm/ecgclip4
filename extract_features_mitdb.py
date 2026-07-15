"""Extract frozen BiomedCLIP image embeddings for MIT-BIH beat images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config_mitdb as C
from model_utils import get_device, load_biomedclip
from zero_shot_eval_mitdb import encode_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    args = parser.parse_args()

    labels = pd.read_csv(
        Path(C.WORK_DIR) / "labels.csv",
        index_col="ecg_id",
    )

    device = get_device()
    model, preprocess, _ = load_biomedclip(
        device,
        ckpt_path=args.ckpt,
    )

    for split_name in ("train", "val", "test"):
        split_df = labels[labels["split"] == split_name]
        if args.limit is not None:
            split_df = split_df.head(args.limit)
        if split_df.empty:
            continue

        print(f"[{split_name}] {len(split_df):,} beats")
        features = encode_images(
            model,
            preprocess,
            device,
            split_df.index.tolist(),
            batch_size=args.batch_size,
        ).numpy().astype(np.float32)

        targets = split_df[C.CLASSES].to_numpy(dtype=np.float32)
        ids = split_df.index.to_numpy(dtype=np.int64)

        np.save(Path(C.FEAT_DIR) / f"X_{split_name}.npy", features)
        np.save(Path(C.FEAT_DIR) / f"y_{split_name}.npy", targets)
        np.save(Path(C.FEAT_DIR) / f"ids_{split_name}.npy", ids)

        print(
            f"saved X_{split_name}.npy {features.shape}, "
            f"y_{split_name}.npy {targets.shape}"
        )


if __name__ == "__main__":
    main()
