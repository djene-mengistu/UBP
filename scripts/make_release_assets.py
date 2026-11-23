#!/usr/bin/env python3
"""Build UBP-only tables and figures from saved experiment records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results"
FIG = OUT / "figures"
DATA = {
    "acdc": {"train_patients": 90, "val_patients": 10, "test_patients": 50,
             "train_slices": 1902, "empty_slices": 61},
    "synapse": {"train_patients": 15, "val_patients": 3, "test_patients": 12,
                "train_slices": 2211, "empty_slices": 931,
                "class_pixels": [922056, 245028, 1344662, 1333973,
                                 14192919, 651861, 3119429, 3474831]},
}
COLORS = {"blue": "#6682A3", "teal": "#08A88A", "ink": "#18324B",
          "muted": "#5B6B7B", "purple": "#7966A8", "red": "#C4514A"}
CLASS_NAMES = {
    "acdc": ["Right ventricle", "Myocardium", "Left ventricle"],
    "synapse": ["Aorta", "Gallbladder", "Left kidney", "Right kidney",
                "Liver", "Pancreas", "Spleen", "Stomach"],
}


def read_json(name: str) -> dict:
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def collect() -> dict:
    records = {}
    for dataset in ("acdc", "synapse"):
        records[dataset] = {
            "test": read_json(f"{dataset}_ubp_test_metrics.json"),
            "validation": read_json(f"{dataset}_ubp_val_metrics.json"),
            "profile": read_json(f"{dataset}_ubp_profile.json"),
            "training": read_json(f"{dataset}_ubp_status.json"),
            "history": [json.loads(line) for line in
                        (RAW / f"{dataset}_ubp_history.jsonl").read_text(encoding="utf-8").splitlines()
                        if line.strip()],
        }
    return records


def make_summary(records: dict) -> dict:
    summary = {
        "title": "UBP preliminary results",
        "seed": 1234,
        "epochs": 300,
        "data_protocol": DATA,
        "research_status": "Single-seed exploratory results; not evidence of robust performance or novelty.",
        "datasets": {},
    }
    for dataset, result in records.items():
        test, val = result["test"], result["validation"]
        per_class = test["per_class"]
        hd_values = [item["mean_hd95"] for item in per_class.values()
                     if item["mean_hd95"] is not None]
        summary["datasets"][dataset] = {
            "split": DATA[dataset],
            "test_patient_mean_dice": test["patient_mean_dice"],
            "test_patient_count": test["patient_count"],
            "test_case_count": test["case_count"],
            "bootstrap_95_ci": test.get("patient_bootstrap_95_ci"),
            "test_mean_foreground_hd95_voxels": float(np.mean(hd_values)),
            "test_per_class": per_class,
            "class_names": {str(index): name for index, name in
                            enumerate(CLASS_NAMES[dataset], start=1)},
            "validation_patient_mean_dice": val["patient_mean_dice"],
            "validation_patient_count": val["patient_count"],
            "validation_role": test["validation_role"],
            "hd95_units": test["hd95_units"],
            "evaluation_throughput_slices_per_second": test.get("inference_throughput_slices_per_second"),
            "profile": result["profile"],
            "training": result["training"],
        }
    return summary


def plot_data_overview() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    ax = axes[0]
    x = np.arange(2)
    width = 0.23
    groups = (("train_patients", "Train", COLORS["ink"]),
              ("val_patients", "Validation", COLORS["teal"]),
              ("test_patients", "Test", COLORS["blue"]))
    for index, (field, label, color) in enumerate(groups):
        values = [DATA[name][field] for name in ("acdc", "synapse")]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        ax.bar_label(bars, padding=2, fontsize=9)
    ax.set_xticks(x, ["ACDC", "Synapse"])
    ax.set_ylabel("Patients / cases")
    ax.set_title("Patient-level experiment split")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    names = ("acdc", "synapse")
    values = [100 * DATA[name]["empty_slices"] / DATA[name]["train_slices"] for name in names]
    bars = ax.bar(["ACDC", "Synapse"], values, color=[COLORS["blue"], COLORS["teal"]], width=0.58)
    labels = [f"{DATA[name]['empty_slices']:,} / {DATA[name]['train_slices']:,}\n{value:.1f}%"
              for name, value in zip(names, values)]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=9)
    ax.set_ylabel("Empty-mask slices (%)")
    ax.set_title("Background-only masks retained")
    ax.set_ylim(0, max(values) * 1.35)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MEDDATA audit and patient-level partition", color=COLORS["ink"], fontsize=14, weight="bold")
    fig.savefig(FIG / "data_overview.png", dpi=190, facecolor="white")
    plt.close(fig)


def plot_per_class(summary: dict, dataset: str, class_count: int) -> None:
    results = summary["datasets"][dataset]["test_per_class"]
    class_ids = np.arange(1, class_count + 1)
    values = [results[str(class_id)]["mean_dice"] for class_id in class_ids]
    fig, ax = plt.subplots(figsize=(max(8.0, class_count * 1.15), 4.6), constrained_layout=True)
    bars = ax.bar(class_ids, values, color=COLORS["teal"], width=0.62)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(class_ids, CLASS_NAMES[dataset], rotation=20 if dataset == "synapse" else 0,
                  ha="right" if dataset == "synapse" else "center")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"Mean Dice across {summary['datasets'][dataset]['test_patient_count']} test patients/cases")
    ax.set_title(f"{dataset.upper()} test: UBP per-class Dice", color=COLORS["ink"], weight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / f"{dataset}_per_class_dice.png", dpi=190, facecolor="white")
    plt.close(fig)


def plot_class_imbalance() -> None:
    counts = DATA["synapse"]["class_pixels"]
    labels = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    bars = ax.barh(labels, counts,
                   color=[COLORS["red"] if class_id == 2 else COLORS["blue"] for class_id in labels])
    ax.set_xscale("log")
    ax.set_yticks(labels, CLASS_NAMES["synapse"])
    ax.invert_yaxis()
    ax.set_xlabel("Foreground pixels in the training pool (log scale)")
    ax.set_title("Synapse foreground class imbalance", color=COLORS["ink"], weight="bold")
    ax.grid(axis="x", alpha=0.2, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts):
        ax.text(count * 1.06, bar.get_y() + bar.get_height() / 2, f"{count:,}", va="center", fontsize=8)
    fig.savefig(FIG / "synapse_class_imbalance.png", dpi=190, facecolor="white")
    plt.close(fig)


def plot_training_curves(records: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), constrained_layout=True)
    for row, dataset in enumerate(("acdc", "synapse")):
        history = records[dataset]["history"]
        for col, field, title, ylabel in (
                (0, "loss", "Training loss", "Loss"),
                (1, "val_mean_dice", "Internal validation Dice", "Patient mean Dice")):
            ax = axes[row, col]
            points = [item for item in history if item.get(field) is not None]
            ax.plot([item["epoch"] for item in points], [item[field] for item in points],
                    color=COLORS["teal"], lw=1.8)
            ax.set_title(f"{dataset.upper()} - {title}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("UBP training history; validation patients held out from the training pool",
                 color=COLORS["ink"], fontsize=13, weight="bold")
    fig.savefig(FIG / "training_curves.png", dpi=190, facecolor="white")
    plt.close(fig)


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.4))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 7)
    ax.axis("off")
    blocks = [
        (0.25, 3.25, 1.35, 1.0, "Image", "224x224", "#EAF1F6"),
        (2.0, 3.25, 1.85, 1.0, "Backbone", "encoder + decoder", "#DCE7F1"),
        (4.3, 3.25, 1.6, 1.0, "Late feature F", "56x56x96", "#DCE7F1"),
        (4.25, 5.45, 1.8, 0.9, "Coarse logits", "class probabilities", "#FFF0D6"),
        (6.85, 4.45, 1.9, 0.9, "Class prototypes", "reliable interiors", "#D9F2EA"),
        (6.85, 2.2, 1.9, 0.9, "Structure + entropy", "edges / uncertainty", "#E7E2F5"),
        (9.4, 3.2, 1.75, 1.05, "Gated refiner", "predict delta", "#FCE4E1"),
        (11.35, 3.2, 1.5, 1.05, "Residual add", "F + alpha*g*delta", "#D9F2EA"),
        (13.15, 3.2, 1.3, 1.05, "Output head", "classifier -> logits", "#DCE7F1"),
    ]
    for x, y, width, height, title, subtitle, face in blocks:
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.08,rounding_size=0.12",
                             linewidth=1.2, edgecolor=COLORS["muted"], facecolor=face)
        ax.add_patch(box)
        ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center",
                fontsize=10, weight="bold", color=COLORS["ink"])
        ax.text(x + width / 2, y + height * 0.30, subtitle, ha="center", va="center",
                fontsize=8.5, color=COLORS["muted"])

    def arrow(start, end, color=COLORS["muted"], rad=0.0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.3, color=color,
                                     connectionstyle=f"arc3,rad={rad}"))

    arrow((1.65, 3.75), (1.95, 3.75))
    arrow((3.9, 3.75), (4.25, 3.75))
    arrow((5.1, 4.3), (5.1, 5.4), color=COLORS["blue"])
    arrow((5.9, 3.9), (6.8, 4.8), color=COLORS["teal"], rad=-0.06)
    arrow((5.9, 5.85), (6.8, 5.0), color=COLORS["blue"], rad=0.05)
    arrow((5.85, 3.5), (6.8, 2.85), color=COLORS["purple"], rad=0.04)
    arrow((5.2, 5.4), (6.8, 2.95), color=COLORS["purple"], rad=-0.05)
    arrow((8.8, 4.85), (9.35, 4.0), color=COLORS["teal"], rad=-0.05)
    arrow((8.8, 2.7), (9.35, 3.45), color=COLORS["purple"], rad=0.05)
    arrow((5.9, 3.55), (9.3, 3.55), color=COLORS["muted"], rad=-0.13)
    arrow((11.2, 3.72), (11.3, 3.72), color=COLORS["teal"])
    arrow((12.9, 3.72), (13.1, 3.72))
    ax.text(7.25, 6.55, "Prediction-derived context and selective residual refinement",
            ha="center", color=COLORS["ink"], fontsize=12, weight="bold")
    ax.text(7.3, 0.65,
            "Training uses auxiliary CE+Dice on coarse logits. Inference uses predictions only; alpha starts at zero.",
            ha="center", fontsize=9, color=COLORS["muted"])
    fig.savefig(FIG / "ubp_architecture.png", dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_tables(summary: dict) -> None:
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows = []
    for dataset, result in summary["datasets"].items():
        profile, training = result["profile"], result["training"]
        rows.append({
            "dataset": dataset,
            "validation_patient_mean_dice": result["validation_patient_mean_dice"],
            "test_patient_mean_dice": result["test_patient_mean_dice"],
            "test_patient_count": result["test_patient_count"],
            "test_case_count": result["test_case_count"],
            "bootstrap_95_ci_low": result["bootstrap_95_ci"][0],
            "bootstrap_95_ci_high": result["bootstrap_95_ci"][1],
            "macro_hd95_voxels": result["test_mean_foreground_hd95_voxels"],
            "parameters": profile["parameters"],
            "partial_scope_flops": profile["flops_estimate"],
            "batch1_median_latency_ms": profile["latency_ms_median"],
            "network_batch1_slices_per_second": profile["throughput_slices_per_second_batch1"],
            "whole_volume_eval_slices_per_second": result["evaluation_throughput_slices_per_second"],
            "training_seconds": training["training_seconds"],
            "peak_training_vram_mib": training["peak_allocated_mib"],
        })
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# UBP experiment report", "",
        "Preliminary single-seed results (seed 1234, 300 epochs). Checkpoints were selected on internal validation patients from the training pool; supplied test partitions were evaluated afterwards.", "",
        "## Aggregate test results", "",
        "| Dataset | Test patients/cases | Patient mean Dice (bootstrap 95% CI) | Macro foreground HD95 (voxels) | Validation mean Dice |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, result in summary["datasets"].items():
        low, high = result["bootstrap_95_ci"]
        lines.append(f"| {dataset.upper()} | {result['test_patient_count']} | "
                     f"{result['test_patient_mean_dice']:.4f} ({low:.4f}-{high:.4f}) | "
                     f"{result['test_mean_foreground_hd95_voxels']:.2f} | "
                     f"{result['validation_patient_mean_dice']:.4f} |")
    for dataset, result in summary["datasets"].items():
        lines += ["", f"## {dataset.upper()} per-class test metrics", "",
                  "| Class | Mean Dice | HD95 (voxels; valid / total) |",
                  "|---|---:|---:|"]
        for class_id, item in result["test_per_class"].items():
            distance = "undefined" if item["mean_hd95"] is None else f"{item['mean_hd95']:.2f}"
            name = CLASS_NAMES[dataset.lower()][int(class_id) - 1]
            lines.append(f"| {name} (ID {class_id}) | {item['mean_dice']:.4f} | "
                         f"{distance} ({item['valid_hd95_count']}/{item['hd95_denominator']}) |")
    lines += ["", "## Compute measurements", "",
              "| Dataset | Parameters | Partial FLOPs (G) | Batch-one median (ms) | Network slices/s | Full-volume slices/s | Peak VRAM (GiB) | Train time (min) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset, result in summary["datasets"].items():
        profile, training = result["profile"], result["training"]
        lines.append(f"| {dataset.upper()} | {profile['parameters']:,} | "
                     f"{profile['flops_estimate']/1e9:.3f} | {profile['latency_ms_median']:.2f} | "
                     f"{profile['throughput_slices_per_second_batch1']:.2f} | "
                     f"{result['evaluation_throughput_slices_per_second']:.2f} | "
                     f"{training['peak_allocated_mib']/1024:.2f} | {training['training_seconds']/60:.1f} |")
    lines += [
        "", "## Notes", "",
        "- Bootstrap intervals resample patients/cases for this single trained seed; they do not capture variation across training seeds.",
        "- HD95 is in voxel units. Physical image spacing was unavailable. Undefined empty/missing-class outcomes are retained in the raw JSON; macro HD95 averages defined class values.",
        "- FLOPs count only profiler-supported convolution/matrix operators. Network latency is warmed-up batch-one; full-volume throughput includes preprocessing and restoration. Shared-GPU measurements are descriptive snapshots.",
        "- Results are exploratory, not a claim of novelty, state-of-the-art performance, or clinical validity.",
        "", "## Raw artifacts", "",
        "Per-case metrics, profiles, status records, and epoch histories are in `raw/`. Run `python scripts/make_release_assets.py` to regenerate this report, the CSV/JSON summary, and aggregate figures.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    records = collect()
    summary = make_summary(records)
    write_tables(summary)
    plot_data_overview()
    plot_per_class(summary, "acdc", 3)
    plot_per_class(summary, "synapse", 8)
    plot_class_imbalance()
    plot_training_curves(records)
    plot_architecture()
    print(f"Wrote UBP report and figures under {OUT}")


if __name__ == "__main__":
    main()
