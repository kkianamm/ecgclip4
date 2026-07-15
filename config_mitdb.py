"""Configuration for using MIT-BIH Arrhythmia Database with ECGClip.

This file is intentionally separate from config.py so the existing PTB-XL
pipeline remains unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

# Point this at the extracted PhysioNet MIT-BIH version root. The folder must
# contain RECORDS and files such as 100.hea, 100.dat, and 100.atr.
DATA_DIR = os.environ.get(
    "MITDB_DATA_DIR",
    "/lambda/nfs/Kiana2/ecgclip/data/mitdb/1.0.0",
)

WORK_DIR = os.environ.get("MITDB_WORK_DIR", "./work_mitdb")
IMG_DIR = os.path.join(WORK_DIR, "images")
FEAT_DIR = os.path.join(WORK_DIR, "features")
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
RESULTS_DIR = os.path.join(WORK_DIR, "results")

for directory in (WORK_DIR, IMG_DIR, FEAT_DIR, CKPT_DIR, RESULTS_DIR):
    Path(directory).mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 360

# AAMI EC57-style beat groups.
CLASSES = ["N", "S", "V", "F", "Q"]

CLASS_DESCRIPTIONS = {
    "N": "normal, bundle branch, or escape beat morphology",
    "S": "supraventricular ectopic beat morphology",
    "V": "ventricular ectopic beat morphology",
    "F": "fusion beat morphology",
    "Q": "paced or unclassifiable beat morphology",
}

PROMPT_TEMPLATES = [
    "an electrocardiogram beat showing {}",
    "a two-lead ECG rhythm strip with {}",
    "an ECG waveform centered on a beat with {}",
    "ECG morphology consistent with {}",
]

BIOMEDCLIP_HF = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
CONTEXT_LENGTH = 256

# labels.csv uses the same fold convention as the PTB-XL repository so the
# downstream split logic remains familiar.
TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4

LP_EPOCHS = 80
LP_LR = 1e-3
LP_WEIGHT_DECAY = 1e-4

FT_EPOCHS = 5
FT_LR = 1e-5
FT_WEIGHT_DECAY = 0.05
