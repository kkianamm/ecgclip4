# MIT-BIH adapter for ECGClip

This adapter keeps the existing PTB-XL pipeline intact and adds a parallel
MIT-BIH beat-classification workflow.

## Why the preprocessing is different

PTB-XL provides 10-second, 12-lead records with record-level diagnostic labels.
MIT-BIH provides 30-minute, two-channel recordings sampled at 360 Hz, with a
symbol attached to each annotated beat. Therefore, this adapter:

1. reads each `.atr` annotation file;
2. maps beat symbols to AAMI-style `N`, `S`, `V`, `F`, and `Q` groups;
3. extracts a beat-centered two-second window;
4. renders both channels as an ECG-paper image;
5. assigns every record to exactly one train/validation/test split.

Record-level splitting is essential. Randomly splitting beats would leak nearly
identical beats from the same patient/record into train and test.

## Copy files

Copy these files into the root of your `ecgclip` repository:

- `config_mitdb.py`
- `mitdb_to_image.py`
- `prepare_mitdb.py`
- `zero_shot_eval_mitdb.py`
- `extract_features_mitdb.py`
- `linear_probe_mitdb.py`
- `finetune_clip_mitdb.py`

The existing `model_utils.py` is reused.

## Dataset layout

After extracting the PhysioNet ZIP, point `MITDB_DATA_DIR` at the folder that
directly contains `RECORDS`, `100.hea`, `100.dat`, and `100.atr`.

Example:

```bash
export MITDB_DATA_DIR=/lambda/nfs/Kiana2/ecgclip/data/mitdb/1.0.0
```

If your ZIP created another nested directory, locate the real root with:

```bash
find /lambda/nfs/Kiana2/ecgclip/data/mitdb -name RECORDS -print
```

## 1. Smoke test

```bash
python prepare_mitdb.py --limit 500
```

This creates:

```text
work_mitdb/
├── labels.csv
└── images/
```

When changing `--limit`, `--max-per-class`, or `--exclude-paced`, rebuild the
metadata:

```bash
rm -rf work_mitdb
python prepare_mitdb.py --limit 500
```

or:

```bash
python prepare_mitdb.py --limit 500 --overwrite-labels
```

## 2. Recommended practical subset

The full database contains roughly 110,000 beat annotations and can create a
large image directory. A balanced cap is usually more practical for the first
experiment:

```bash
python prepare_mitdb.py --max-per-class 5000
```

To use the stricter 44-record AAMI inter-patient protocol, which omits the four
paced records:

```bash
python prepare_mitdb.py --exclude-paced --max-per-class 5000
```

Note that omitting paced records can leave the `Q` class extremely small.

## 3. Zero-shot evaluation

```bash
python zero_shot_eval_mitdb.py
```

Quick test:

```bash
python zero_shot_eval_mitdb.py --limit 500
```

## 4. Frozen BiomedCLIP features and linear probe

```bash
python extract_features_mitdb.py
python linear_probe_mitdb.py
```

## 5. Fine-tune the vision tower

```bash
python finetune_clip_mitdb.py --epochs 5
python zero_shot_eval_mitdb.py \
  --ckpt work_mitdb/checkpoints/biomedclip_mitdb_ft.pt
```

The fine-tuning loss uses one text prototype per beat class. This avoids the
ordinary CLIP InfoNCE problem where two images with the same class caption are
incorrectly treated as negatives.

## Useful environment variables

```bash
export MITDB_DATA_DIR=/path/to/mitdb/1.0.0
export MITDB_WORK_DIR=./work_mitdb
```

## AAMI mapping used

```text
N: N, L, R, e, j
S: A, a, J, S
V: V, E
F: F
Q: /, f, Q
```

Non-beat annotation symbols are ignored.
