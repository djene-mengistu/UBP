#!/usr/bin/env python3
"""Save a side-by-side image, ground-truth, and prediction overlay figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="2D grayscale image as .npy")
    parser.add_argument("--label", type=Path, required=True, help="2D integer ground-truth map as .npy")
    parser.add_argument("--prediction", type=Path, required=True, help="2D integer prediction map as .npy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image, label, prediction = (np.load(path, allow_pickle=False) for path in
                                (args.image, args.label, args.prediction))
    if image.ndim != 2 or label.shape != image.shape or prediction.shape != image.shape:
        parser.error("image, label, and prediction must be 2D arrays with equal shapes")
    if not np.isfinite(image).all() or not np.isfinite(label).all() or not np.isfinite(prediction).all():
        parser.error("arrays must contain finite values")
    if not np.equal(label, np.floor(label)).all() or not np.equal(prediction, np.floor(prediction)).all():
        parser.error("label and prediction arrays must contain integer class IDs")

    classes = max(int(label.max()), int(prediction.max())) + 1
    palette = plt.get_cmap("tab10", max(classes, 2))
    cmap = palette.copy()
    cmap.set_under((0, 0, 0, 0))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")
    for axis, mask, title in ((axes[1], label, "Ground truth"),
                              (axes[2], prediction, "Prediction")):
        axis.imshow(image, cmap="gray")
        overlay = np.ma.masked_where(mask == 0, mask)
        axis.imshow(overlay, cmap=cmap, vmin=1, vmax=max(classes - 1, 1), alpha=0.48,
                    interpolation="nearest")
        axis.set_title(title)
    for axis in axes:
        axis.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
