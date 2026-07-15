"""Render a beat-centered, two-channel MIT-BIH ECG window as an image."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _safe_lead_names(lead_names: Sequence[str] | None, n_leads: int) -> list[str]:
    if lead_names is None:
        return [f"lead {i + 1}" for i in range(n_leads)]
    names = [str(name) for name in lead_names]
    if len(names) < n_leads:
        names.extend(f"lead {i + 1}" for i in range(len(names), n_leads))
    return names[:n_leads]


def render_mitdb_beat(
    signal: np.ndarray,
    fs: float = 360.0,
    lead_names: Sequence[str] | None = None,
    *,
    style: str = "grid",
    dpi: int = 120,
    line_width: float = 0.9,
    r_peak_index: int | None = None,
) -> Image.Image:
    """Return an RGB image for a beat-centered ECG window.

    Parameters
    ----------
    signal:
        Array shaped ``(n_samples, n_leads)`` in millivolts.
    fs:
        Sampling rate in Hz.
    lead_names:
        Names from the WFDB header, normally MLII and a precordial lead.
    style:
        ``grid`` for ECG-paper styling or ``plain`` for a white background.
    r_peak_index:
        Optional R-peak location inside the extracted window. A subtle
        center marker is drawn when provided.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2 or signal.shape[0] < 2:
        raise ValueError(f"Expected (samples, leads), got {signal.shape}")

    n_samples, n_leads = signal.shape
    names = _safe_lead_names(lead_names, n_leads)
    t = np.arange(n_samples, dtype=np.float32) / float(fs)
    duration = n_samples / float(fs)

    fig_height = max(2.2, 1.65 * n_leads)
    fig, axes = plt.subplots(
        n_leads,
        1,
        figsize=(7.0, fig_height),
        dpi=dpi,
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    fig.subplots_adjust(
        hspace=0.04,
        left=0.10,
        right=0.99,
        top=0.97,
        bottom=0.14,
    )

    finite = signal[np.isfinite(signal)]
    robust_amp = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
    robust_amp = min(max(robust_amp * 1.15, 0.75), 5.0)

    for lead_idx, ax in enumerate(axes):
        ax.plot(t, signal[:, lead_idx], color="black", linewidth=line_width)
        ax.set_xlim(0.0, duration)
        ax.set_ylim(-robust_amp, robust_amp)

        if style == "grid":
            ax.set_facecolor("#fff7f7")
            ax.set_xticks(np.arange(0, duration + 1e-6, 0.2), minor=True)
            ax.set_xticks(np.arange(0, duration + 1e-6, 1.0), minor=False)
            ax.set_yticks(np.arange(-robust_amp, robust_amp + 1e-6, 0.1), minor=True)
            ax.set_yticks(np.arange(-robust_amp, robust_amp + 1e-6, 0.5), minor=False)
            ax.grid(which="minor", color="#f2bcbc", linewidth=0.28)
            ax.grid(which="major", color="#dc7f7f", linewidth=0.55)
        elif style == "plain":
            ax.set_facecolor("white")
        else:
            raise ValueError("style must be 'grid' or 'plain'")

        if r_peak_index is not None:
            r_time = float(r_peak_index) / float(fs)
            ax.axvline(r_time, color="black", alpha=0.18, linewidth=0.6)

        ax.set_ylabel(names[lead_idx], rotation=0, labelpad=28, va="center", fontsize=8)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.tick_params(axis="x", which="both", labelsize=7, length=0)

        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("time (s)", fontsize=8)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    buffer.close()
    return image


def render_to_file(
    signal: np.ndarray,
    fs: float,
    out_path: str | Path,
    lead_names: Sequence[str] | None = None,
    **kwargs,
) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = render_mitdb_beat(
        signal,
        fs=fs,
        lead_names=lead_names,
        **kwargs,
    )
    image.save(out_path)
    return str(out_path)
