"""Prepare MIT-BIH beat images and metadata for the ECGClip repository.

The PTB-XL pipeline classifies complete 10-second, 12-lead records. MIT-BIH is
different: it contains 30-minute, two-channel recordings with beat annotations.
This adapter creates one beat-centered image per selected annotation and maps
the original symbols to AAMI-style N/S/V/F/Q classes.

Expected data root
------------------
The folder supplied by --data-dir must contain files such as:

    RECORDS
    100.hea
    100.dat
    100.atr

Examples
--------
Smoke test:
    python prepare_mitdb.py --limit 500

Metadata only:
    python prepare_mitdb.py --no-render

Practical balanced subset:
    python prepare_mitdb.py --max-per-class 5000

Full mapped dataset:
    python prepare_mitdb.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

import config_mitdb as C
from mitdb_to_image import render_to_file


# Common AAMI EC57 mapping used for MIT-BIH inter-patient beat classification.
AAMI_MAP = {
    # N: normal, bundle branch, atrial/nodal escape
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # S: supraventricular ectopic
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # V: ventricular ectopic
    "V": "V",
    "E": "V",
    # F: fusion of ventricular and normal
    "F": "F",
    # Q: paced, fusion of paced and normal, or unclassifiable
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

# Widely used inter-patient split from the MIT-BIH literature.
AAMI_DS1 = {
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
}
AAMI_DS2 = {
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
}

# Five DS1 records are held out as validation. Splitting is always by record,
# never by beat, to prevent patient/record leakage.
VALIDATION_RECORDS = {"109", "114", "118", "201", "207"}

# The four commonly excluded paced records. By default this adapter includes
# them so the Q class has useful examples. Use --exclude-paced for the strict
# 44-record inter-patient protocol.
PACED_RECORDS = {"102", "104", "107", "217"}


def discover_records(data_dir: str | Path) -> list[str]:
    root = Path(data_dir).expanduser().resolve()
    records_file = root / "RECORDS"

    if records_file.exists():
        records = [
            line.strip()
            for line in records_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        records = sorted(path.stem for path in root.glob("*.hea"))

    records = [record for record in records if (root / f"{record}.hea").exists()]
    if not records:
        raise FileNotFoundError(
            f"No MIT-BIH WFDB records found below {root}. "
            "Point --data-dir at the version root containing RECORDS and 100.hea."
        )
    return sorted(set(records))


def split_for_record(record_id: str, exclude_paced: bool) -> str | None:
    if exclude_paced and record_id in PACED_RECORDS:
        return None

    if record_id in AAMI_DS2:
        return "test"
    if record_id in VALIDATION_RECORDS:
        return "val"
    if record_id in AAMI_DS1:
        return "train"

    # Keep the paced records patient-disjoint while giving Q examples to train
    # and test. Any future extra records are deterministically placed in train.
    if record_id == "217":
        return "test"
    if record_id in {"102", "104", "107"}:
        return "train"
    return "train"


def fold_for_split(split: str) -> int:
    return {"train": 1, "val": C.VAL_FOLD, "test": C.TEST_FOLD}[split]


def image_path_for(ecg_id: int) -> str:
    return os.path.join(C.IMG_DIR, f"{int(ecg_id):07d}.png")


def build_metadata(
    data_dir: str | Path,
    records: Iterable[str],
    *,
    exclude_paced: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    next_id = 1
    root = Path(data_dir)

    for record_id in tqdm(list(records), desc="Reading MIT-BIH annotations"):
        split = split_for_record(record_id, exclude_paced=exclude_paced)
        if split is None:
            continue

        record_path = str(root / record_id)
        annotation = wfdb.rdann(record_path, "atr")

        for annotation_index, (sample, symbol) in enumerate(
            zip(annotation.sample, annotation.symbol)
        ):
            aami_class = AAMI_MAP.get(symbol)
            if aami_class is None:
                continue

            description = C.CLASS_DESCRIPTIONS[aami_class]
            row = {
                "ecg_id": next_id,
                "record_id": record_id,
                "patient_id": record_id,
                "annotation_index": int(annotation_index),
                "r_peak_sample": int(sample),
                "beat_symbol": str(symbol),
                "aami_class": aami_class,
                "split": split,
                "strat_fold": fold_for_split(split),
                "superclasses": aami_class,
                "report": f"a two-lead ECG beat showing {description}",
            }
            for class_name in C.CLASSES:
                row[class_name] = float(aami_class == class_name)
            rows.append(row)
            next_id += 1

    if not rows:
        raise RuntimeError("No mapped beat annotations were found.")

    dataframe = pd.DataFrame(rows).set_index("ecg_id")
    return dataframe


def sample_metadata(
    dataframe: pd.DataFrame,
    *,
    max_per_class: int | None,
    limit: int | None,
    seed: int,
) -> pd.DataFrame:
    sampled = dataframe

    if max_per_class is not None:
        pieces = []
        for split_name in ("train", "val", "test"):
            split_df = sampled[sampled["split"] == split_name]
            for class_name in C.CLASSES:
                class_df = split_df[split_df["aami_class"] == class_name]
                if len(class_df) > max_per_class:
                    class_df = class_df.sample(
                        n=max_per_class,
                        random_state=seed,
                    )
                pieces.append(class_df)
        sampled = pd.concat(pieces, axis=0)

    sampled = sampled.sort_values(
        ["record_id", "r_peak_sample", "annotation_index"]
    )

    if limit is not None:
        # A deterministic stratified-ish smoke-test subset across split/class.
        groups = []
        group_count = max(1, sampled.groupby(["split", "aami_class"]).ngroups)
        per_group = max(1, int(np.ceil(limit / group_count)))
        for _, group in sampled.groupby(["split", "aami_class"], sort=True):
            groups.append(group.head(per_group))
        sampled = pd.concat(groups).head(limit)

    # Reindex after sampling so image IDs are compact and deterministic.
    sampled = sampled.reset_index(drop=True)
    sampled.index = np.arange(1, len(sampled) + 1)
    sampled.index.name = "ecg_id"
    return sampled


def extract_window(
    signal: np.ndarray,
    center_sample: int,
    window_samples: int,
) -> tuple[np.ndarray, int]:
    half_left = window_samples // 2
    half_right = window_samples - half_left
    start = center_sample - half_left
    stop = center_sample + half_right

    pad_left = max(0, -start)
    pad_right = max(0, stop - len(signal))
    clipped_start = max(0, start)
    clipped_stop = min(len(signal), stop)

    window = signal[clipped_start:clipped_stop]
    if pad_left or pad_right:
        window = np.pad(
            window,
            ((pad_left, pad_right), (0, 0)),
            mode="edge",
        )

    if len(window) != window_samples:
        raise RuntimeError(
            f"Window extraction produced {len(window)} samples; "
            f"expected {window_samples}"
        )

    r_peak_in_window = half_left
    return window.astype(np.float32), r_peak_in_window


def render_all(
    dataframe: pd.DataFrame,
    data_dir: str | Path,
    *,
    window_seconds: float,
    style: str,
) -> None:
    root = Path(data_dir)

    for record_id, record_rows in tqdm(
        dataframe.groupby("record_id", sort=True),
        desc="Rendering records",
    ):
        record_path = str(root / str(record_id))
        signal, fields = wfdb.rdsamp(record_path)
        signal = signal.astype(np.float32)
        fs = float(fields.get("fs", C.SAMPLING_RATE))
        lead_names = fields.get("sig_name")
        window_samples = max(2, int(round(window_seconds * fs)))

        for ecg_id, row in record_rows.iterrows():
            out_path = image_path_for(int(ecg_id))
            if os.path.exists(out_path):
                continue

            window, r_peak_in_window = extract_window(
                signal,
                center_sample=int(row["r_peak_sample"]),
                window_samples=window_samples,
            )
            render_to_file(
                window,
                fs,
                out_path,
                lead_names=lead_names,
                style=style,
                r_peak_index=r_peak_in_window,
            )


def print_summary(dataframe: pd.DataFrame) -> None:
    print(f"Prepared beats: {len(dataframe):,}")
    print("\nSplit counts:")
    print(dataframe["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int))
    print("\nClass counts by split:")
    table = pd.crosstab(dataframe["split"], dataframe["aami_class"])
    print(table.reindex(index=["train", "val", "test"], columns=C.CLASSES, fill_value=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=C.DATA_DIR)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--style", choices=["grid", "plain"], default="grid")
    parser.add_argument(
        "--exclude-paced",
        action="store_true",
        help="Use the strict 44-record AAMI split and omit 102/104/107/217.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional maximum beats per split and class.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Small deterministic total subset for a smoke test.",
    )
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--overwrite-labels",
        action="store_true",
        help="Rebuild labels.csv even when it already exists.",
    )
    args = parser.parse_args()

    labels_path = Path(C.WORK_DIR) / "labels.csv"

    if labels_path.exists() and not args.overwrite_labels:
        dataframe = pd.read_csv(labels_path, index_col="ecg_id")
        print(f"Using existing metadata: {labels_path}")
    else:
        records = discover_records(args.data_dir)
        print(f"Found {len(records)} records below {Path(args.data_dir).resolve()}")
        dataframe = build_metadata(
            args.data_dir,
            records,
            exclude_paced=args.exclude_paced,
        )
        dataframe = sample_metadata(
            dataframe,
            max_per_class=args.max_per_class,
            limit=args.limit,
            seed=args.seed,
        )
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(labels_path)
        print(f"Saved metadata -> {labels_path}")

    print_summary(dataframe)

    if not args.no_render:
        render_all(
            dataframe,
            args.data_dir,
            window_seconds=args.window_seconds,
            style=args.style,
        )
        print(f"Images -> {C.IMG_DIR}")


if __name__ == "__main__":
    main()
