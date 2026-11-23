"""Train, validate, evaluate, and profile UBP on one explicit GPU."""

import argparse
import csv
import json
import os
import random
import subprocess
import time
import importlib.metadata
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (CLASS_NAMES, SPECS, SliceDataset, VolumeRecord, patient_id, read_pair,
                   validate_pair, volume_paths, write_new_json)
from .metrics import aggregate_metrics, segmentation_metrics
from .backbone import build_backbone
from .ubp import build_ubp


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def build_model(config):
    if config.get("architecture") != "ubp":
        raise ValueError("This release supports the UBP architecture only")
    backbone = build_backbone(config["dataset"], image_size=config["image_size"])
    return build_ubp(
        backbone, SPECS[config["dataset"]]["classes"],
        auxiliary_weight=config.get("lambda_aux", 0.3),
        enabled=config.get("refinement_enabled", True),
        selective_gating=config.get("selective_gating", True))


def segmentation_loss(logits, labels, config):
    classes = SPECS[config["dataset"]]["classes"]
    one_hot = torch.nn.functional.one_hot(labels.long(), classes).permute(0, 3, 1, 2).float()
    probabilities = logits.float().softmax(1)
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.square().sum(dims) + one_hot.square().sum(dims)
    class_dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
    dice_loss = 1 - class_dice.mean()  # Average over all classes, including background.
    ce_loss = torch.nn.functional.cross_entropy(logits.float(), labels.long())
    return config["ce_weight"] * ce_loss + config["dice_weight"] * dice_loss


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".writing.pt")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_json(path, value):
    temporary = path.with_suffix(".writing.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


@torch.inference_mode()
def predict_volume(model, image, config, device):
    """Resize slices, infer bounded batches, restore each original label grid."""
    size = config["image_size"]
    result = np.empty(image.shape, dtype=np.uint8)
    interpolation = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}.get(
        config.get("image_interpolation", "linear"))
    if interpolation is None:
        raise ValueError("image_interpolation must be 'linear' or 'cubic'")
    for start in range(0, len(image), config["inference_batch_size"]):
        slices = image[start:start + config["inference_batch_size"]]
        batch = np.stack([cv2.resize(s.astype(np.float32), (size, size), interpolation=interpolation)
                          for s in slices])
        tensor = torch.from_numpy(batch[:, None]).to(device)
        with torch.autocast("cuda", enabled=config["amp"]):
            logits = model(tensor)
        # Restore logits before argmax to avoid coarse nearest-label stair steps.
        logits = torch.nn.functional.interpolate(logits.float(), size=image.shape[-2:],
                                                  mode="bilinear", align_corners=False)
        result[start:start + len(slices)] = logits.argmax(1).cpu().numpy().astype(np.uint8)
    return result


def qualitative_figure(image, label, prediction, destination, num_classes, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colors = ["#111827", "#ef4444", "#22c55e", "#3b82f6", "#f59e0b",
              "#a855f7", "#06b6d4", "#ec4899", "#84cc16"]
    cmap = ListedColormap(colors[:num_classes])
    foreground_slices = np.flatnonzero(np.any(label != 0, axis=(1, 2)))
    if foreground_slices.size == 0:
        return
    indices = np.unique(np.linspace(0, len(foreground_slices) - 1,
                                   min(5, len(foreground_slices)), dtype=int))
    indices = foreground_slices[indices]
    fig, axes = plt.subplots(4, len(indices), figsize=(3.2 * len(indices), 10), squeeze=False)
    for col, z in enumerate(indices):
        for row in range(4):
            ax = axes[row, col]
            ax.imshow(image[z], cmap="gray", interpolation="nearest")
            if row in (1, 2):
                mask = label[z] if row == 1 else prediction[z]
                ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap,
                          vmin=0, vmax=num_classes - 1, alpha=0.65, interpolation="nearest")
            if row == 3:
                error = prediction[z] != label[z]
                ax.imshow(np.ma.masked_where(~error, error), cmap="autumn", vmin=0, vmax=1, alpha=0.8)
            ax.set_title(f"{['Image', 'Ground truth', 'Prediction', 'Error'][row]} | z={z}", fontsize=10)
            ax.axis("off")
    fig.suptitle(title + "\nGround-truth-positive slices; yellow marks any label disagreement", fontsize=14)
    dataset = title.split(" | ", 1)[0].lower()
    names = CLASS_NAMES[dataset]
    fig.legend(handles=[Patch(color=colors[k], label=names[k]) for k in range(1, num_classes)],
               loc="lower center", ncol=min(num_classes - 1, 4), frameon=False)
    fig.subplots_adjust(top=0.9, bottom=0.07, wspace=0.03, hspace=0.18)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def evaluate(model, config, split, partition, device, output=None, hd95=False):
    model.eval()
    spec = SPECS[config["dataset"]]
    all_metrics, cases, by_patient = [], [], defaultdict(list)
    started = time.perf_counter()
    inference_seconds = 0.0
    for i, item in enumerate(volume_paths(config["data_root"], config["dataset"], split, partition, config)):
        path = item.path if isinstance(item, VolumeRecord) else item
        image, label = read_pair(item)
        validate_pair(image, label, spec["classes"])
        inference_started = time.perf_counter()
        prediction = predict_volume(model, image, config, device)
        inference_seconds += time.perf_counter() - inference_started
        metrics = segmentation_metrics(prediction, label, spec["classes"], compute_hd95=hd95)
        all_metrics.append(metrics)
        case_dice = float(np.mean([m["dice"] for m in metrics]))
        patient = item.patient if isinstance(item, VolumeRecord) else patient_id(path)
        by_patient[patient].append(case_dice)
        cases.append({"case": path.stem, "patient": patient, "shape": list(label.shape),
                      "mean_dice": case_dice, "classes": metrics})
        if output is not None:
            print(json.dumps({"event": "evaluated_case", "case": path.name, "dice": case_dice}), flush=True)
            if i < 3:
                qualitative_figure(image, label, prediction, output / f"{partition}_{path.stem}.png",
                                   spec["classes"], f"{config['dataset'].upper()} | {partition} | {path.stem}")
    summary = aggregate_metrics(all_metrics, spec["classes"])
    patient_scores = [float(np.mean(v)) for v in by_patient.values()]
    if not patient_scores:
        raise ValueError("Empty evaluation partition")
    generator = np.random.default_rng(2026)
    bootstrap = generator.choice(patient_scores, size=(2000, len(patient_scores)), replace=True).mean(1)
    summary.update({"partition": partition, "hd95_units": "voxels",
                    "validation_role": split.get("validation_role", "independent_holdout"),
                    "patient_count": len(patient_scores), "patient_mean_dice": float(np.mean(patient_scores)),
                    "patient_bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                    "bootstrap_note": "Patient resampling for this seed; not training-seed uncertainty.",
                    "evaluation_seconds": time.perf_counter() - started,
                    "preprocessing_network_restore_seconds": inference_seconds,
                    "inference_throughput_slices_per_second": sum(c["shape"][0] for c in cases) / inference_seconds,
                    "geometry_policy": (f"{config.get('image_interpolation', 'linear')} image resize to model input; "
                                        "bilinear logit restoration before argmax"),
                    "cases": cases})
    if output is not None:
        write_new_json(output / "metrics.json", summary)
        with (output / "per_case_class.csv").open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "patient", "class_id", "dice", "hd95", "hd95_status", "gt_present", "pred_present"])
            writer.writeheader()
            for case in cases:
                for row in case["classes"]:
                    writer.writerow({"case": case["case"], "patient": case["patient"], **row})
    return summary


@torch.inference_mode()
def profile(model, config, device):
    """Analytic operation estimate plus measured warmed-up batch-one latency."""
    model.eval()
    if config.get("architecture") == "ubp":
        counts = {"conv_linear_macs": 0}
        handles = []

        def count_conv(module, inputs, output):
            counts["conv_linear_macs"] += output.numel() * (module.in_channels // module.groups) * int(np.prod(module.kernel_size))

        def count_linear(module, inputs, output):
            counts["conv_linear_macs"] += output.numel() * module.in_features

        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                handles.append(module.register_forward_hook(count_conv))
            elif isinstance(module, torch.nn.Linear):
                handles.append(module.register_forward_hook(count_linear))
        sample = torch.zeros(1, 1, config["image_size"], config["image_size"], device=device)
        with torch.autocast("cuda", enabled=config["amp"]):
            model(sample)
        for handle in handles:
            handle.remove()
        profiler_flops = None
        profiler_flops_scope = "unavailable"
        try:
            activities = [torch.profiler.ProfilerActivity.CPU,
                          torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities, with_flops=True) as trace:
                with torch.autocast("cuda", enabled=config["amp"]):
                    model(sample)
            torch.cuda.synchronize(device)
            profiler_flops = int(sum(event.flops or 0 for event in trace.key_averages()))
            profiler_flops_scope = (
                "PyTorch profiler-supported convolution/matmul operators only; "
                "does not count all elementwise, normalization, softmax, or custom operations"
            )
        except (RuntimeError, AttributeError, NotImplementedError):
            profiler_flops = None
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast("cuda", enabled=config["amp"]):
            for _ in range(3):
                model(sample)
            torch.cuda.synchronize(device)
            times = []
            for _ in range(10):
                before = time.perf_counter()
                model(sample)
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - before)
        return {"parameters": sum(p.numel() for p in model.parameters()),
                "input_shape": list(sample.shape), "conv_linear_macs": int(counts["conv_linear_macs"]),
                "flops_estimate": profiler_flops,
                "flops_scope": profiler_flops_scope,
                "latency_ms_median": float(np.median(times) * 1000),
                "latency_ms_p95": float(np.quantile(times, .95) * 1000),
                "throughput_slices_per_second_batch1": float(1 / np.mean(times)),
                "profile_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                "precision": "float16_autocast" if config["amp"] else "float32",
                "gpu": torch.cuda.get_device_name(device), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "timing_scope": "network only, warmed up; shared GPU contention may affect latency"}
    counts = {"macs": 0, "bias_additions": 0, "group_norm_flops_approx": 0}
    handles = []

    def count_conv(module, inputs, output):
        counts["macs"] += output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size).item()
        if module.bias is not None:
            counts["bias_additions"] += output.numel()

    def count_norm(module, inputs, output):
        counts["group_norm_flops_approx"] += 5 * output.numel()

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(count_conv))
        if isinstance(module, torch.nn.GroupNorm):
            handles.append(module.register_forward_hook(count_norm))
    size = config["image_size"]
    sample = torch.zeros(1, 1, size, size, device=device)
    with torch.autocast("cuda", enabled=config["amp"]):
        model(sample)
    for handle in handles:
        handle.remove()
    # Four bilinear decoder upsamplings; seven arithmetic ops per output sample.
    interpolation = sum(7 * (config["width"] * 2**(i + 1)) * (size // 2**i)**2 for i in range(4))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", enabled=config["amp"]):
        for _ in range(10):
            model(sample)
        torch.cuda.synchronize(device)
        times = []
        for _ in range(30):
            before = time.perf_counter()
            model(sample)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - before)
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "input_shape": [1, 1, size, size], "conv_macs": int(counts["macs"]),
            "flops_estimate": int(2 * counts["macs"] + counts["bias_additions"] + counts["group_norm_flops_approx"] + interpolation),
            "flops_convention": "2 per MAC; conv bias + approximate GN(5/element) and bilinear(7/output); excludes comparisons/memory ops",
            "latency_ms_median": float(np.median(times) * 1000),
            "latency_ms_p95": float(np.quantile(times, .95) * 1000),
            "throughput_slices_per_second_batch1": float(1 / np.mean(times)),
            "profile_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "precision": "float16_autocast" if config["amp"] else "float32",
            "gpu": torch.cuda.get_device_name(device), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "timing_scope": "network only, warmed up; shared GPU contention may affect latency"}


def plot_history(history_path, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    rows = list({r["epoch"]: r for r in rows}.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot([r["epoch"] for r in rows], [r["loss"] for r in rows], color="#2563eb")
    axes[0].set(xlabel="Epoch", ylabel="Dice + CE loss", title="Training objective")
    val = [r for r in rows if r["val_mean_dice"] is not None]
    axes[1].plot([r["epoch"] for r in val], [r["val_mean_dice"] for r in val], "o-", color="#0f766e")
    axes[1].set(xlabel="Epoch", ylabel="Mean foreground Dice", title="Patient-held-out validation", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def train(model, config, split, device, run, resume=False):
    dataset = SliceDataset(config["data_root"], config["dataset"], split["train"],
                           config["image_size"], cache=config["cache_in_ram"],
                           files=split.get("train_slice_files"), config=config)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        num_workers=config["workers"], pin_memory=True, worker_init_fn=seed_worker,
                        generator=generator, persistent_workers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = config["epochs"] * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / total_steps) ** 0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
    start, best, elapsed = 0, -1.0, 0.0
    if resume:
        checkpoint = torch.load(run / "latest.pt", map_location=device, weights_only=False)
        if checkpoint["config"] != config or checkpoint["split_manifest"] != split:
            raise ValueError("Resume config/split mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, elapsed = checkpoint["epoch"], checkpoint["best_dice"], checkpoint["training_seconds"]
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        torch.cuda.set_rng_state(checkpoint["rng_cuda"].cpu(), device)
        generator.set_state(checkpoint["rng_loader"].cpu())
    print(json.dumps({"event": "training_start", "samples": len(dataset), "start_epoch": start,
                      "train_patients": len(split["train"]), "val_patients": len(split["val"])}), flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start + 1, config["epochs"] + 1):
        before = time.perf_counter()
        model.train()
        loss_sum, n, skipped_updates = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config["amp"]):
                logits = model(images)
                loss = segmentation_loss(logits, labels, config)
                auxiliary_getter = getattr(model, "auxiliary_logits", None)
                auxiliary = auxiliary_getter() if auxiliary_getter is not None else None
                if auxiliary is not None:
                    aux_target = torch.nn.functional.interpolate(
                        labels[:, None].float(), size=auxiliary.shape[-2:], mode="nearest")[:, 0].long()
                    loss = loss + config.get("lambda_aux", 0.3) * segmentation_loss(auxiliary, aux_target, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_norm = config.get("gradient_clip_norm")
            if clip_norm is None:
                # An infinite limit performs a finite-norm check without altering
                # Check finite gradients without changing them.
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                               error_if_nonfinite=not config["amp"])
            else:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm),
                                                      error_if_nonfinite=not config["amp"])
                skipped_updates += int(not torch.isfinite(norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(images)
            n += len(images)
        val_dice = None
        improved = False
        if epoch == 1 or epoch % config["validation_interval"] == 0 or epoch == config["epochs"]:
            validation = evaluate(model, config, split, "val", device)
            val_dice = validation["patient_mean_dice"]
            improved = val_dice > best
            best = max(best, val_dice)
        elapsed += time.perf_counter() - before
        row = {"epoch": epoch, "loss": loss_sum / n, "val_mean_dice": val_dice,
               "best_val_dice": best, "lr": scheduler.get_last_lr()[0],
               "amp_skipped_updates": skipped_updates,
               "epoch_seconds": time.perf_counter() - before, "training_seconds": elapsed,
               "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2}
        with (run / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps({"event": "epoch", **row}), flush=True)
        if improved or epoch % config["validation_interval"] == 0 or epoch == config["epochs"]:
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                          "epoch": epoch, "best_dice": best, "training_seconds": elapsed,
                          "config": config, "split_manifest": split,
                          "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                          "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(device),
                          "rng_loader": generator.get_state()}
            atomic_checkpoint(run / "latest.pt", checkpoint)
            if improved:
                atomic_checkpoint(run / "best_val.pt", checkpoint)
            plot_history(run / "history.jsonl", run / "training_curves.png")
        save_json(run / "status.json", {"phase": "training", **row})
    save_json(run / "status.json", {"phase": "complete", "epochs": config["epochs"],
                                     "best_val_dice": best, "training_seconds": elapsed,
                                     "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "evaluate", "smoke"], default="train")
    parser.add_argument("--partition", choices=["val", "test"], default="val")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partition-manifest", type=Path,
                        help="Explicit existing partition manifest; automatic holdouts are disabled")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.partition_manifest is None:
        parser.error("Automatic re-splitting is disabled. Supply --partition-manifest from the verified existing dataset partitions.")
    split = json.loads(args.partition_manifest.read_text())
    for name in ("train", "val", "test"):
        if not isinstance(split.get(name), list) or not split[name] or len(split[name]) != len(set(split[name])):
            parser.error(f"Invalid/missing explicit {name} partition")
    pairs = [("train", "val"), ("train", "test")]
    shared_test = split.get("validation_role") == "provided_test_used_as_validation"
    if shared_test and set(split["val"]) != set(split["test"]):
        parser.error("Provided-test validation must use exactly the supplied test membership")
    if not shared_test:
        pairs.append(("val", "test"))
    if any(set(split[a]) & set(split[b]) for a, b in pairs):
        parser.error("Explicit partitions overlap; no automatic substitute will be created")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        parser.error("Set exactly one user-authorized GPU in CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(config["threads"])
    cv2.setNumThreads(0)
    if not torch.cuda.is_available():
        raise RuntimeError("Authorized CUDA device unavailable; no automatic fallback")
    device = torch.device("cuda:0")  # Logical zero inside the explicit physical-device mask.
    torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device)
    seed_everything(config["seed"])
    model = build_model(config).to(device)
    run = args.run_dir
    if args.mode == "smoke":
        smoke_size = config["image_size"]
        batch = SliceDataset(config["data_root"], config["dataset"], split["train"], size=smoke_size,
                             cache=False, files=split.get("train_slice_files"), config=config)
        image, label = batch[0]
        for index in range(min(len(batch), 64)):
            image, label = batch[index]
            if label.any():
                break
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        losses = []
        smoke_steps = 2
        for _ in range(smoke_steps):
            optimizer.zero_grad()
            loss = segmentation_loss(model(image[None].to(device)), label[None].to(device), config)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if not np.isfinite(losses).all():
            raise RuntimeError(f"Smoke overfit failed: {losses}")
        print(json.dumps({"smoke": "passed", "loss_start": losses[0], "loss_end": losses[-1],
                          "profile": profile(model, config, device)}), flush=True)
        return
    if args.mode == "evaluate":
        checkpoint = torch.load(run / "best_val.pt", map_location=device, weights_only=False)
        if checkpoint["config"] != config or checkpoint["split_manifest"] != split:
            raise ValueError("Evaluation config/split mismatch")
        model.load_state_dict(checkpoint["model"])
        output = run / ("evaluation_" + args.partition)
        output.mkdir(exist_ok=False)
        summary = evaluate(model, config, split, args.partition, device, output=output, hd95=True)
        write_new_json(output / "profile.json", profile(model, config, device))
        print(json.dumps({"event": "evaluation_complete", "partition": args.partition,
                          "mean_dice": summary["patient_mean_dice"], "output": str(output)}), flush=True)
        return
    if not args.resume:
        run.mkdir(parents=True, exist_ok=False)
        write_new_json(run / "config.json", config)
        write_new_json(run / "split.json", split)
        packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                          if d.metadata['Name'])
        (run / "environment.txt").write_text("\n".join(packages) + "\n")
        write_new_json(run / "runtime.json", {"python": os.sys.version, "executable": os.sys.executable,
                       "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                       "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(device),
                       "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                       "source_revision": json.loads(Path(".source_revision.json").read_text())
                         if Path(".source_revision.json").exists() else None})
        write_new_json(run / "profile_initial.json", profile(model, config, device))
    train(model, config, split, device, run, args.resume)


if __name__ == "__main__":
    main()
