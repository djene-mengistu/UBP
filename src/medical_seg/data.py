"""Read immutable ACDC/Synapse data and freeze patient-level partitions."""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SPECS = {
    "acdc": {"classes": 4, "slices": "ACDC_preprocessed/ACDC_training_slices",
             "volumes": "ACDC_preprocessed/ACDC_training_volumes",
             "test": "ACDC_preprocessed/ACDC_testing_volumes", "extension": ".h5", "val_count": 10},
    "synapse": {"classes": 9, "slices": "Synapse/train_npz",
                "volumes": "Synapse/train_vol_h5", "test": "Synapse/test_vol_h5",
                "extension": ".npz", "val_count": 3},
}

CLASS_NAMES = {
    "acdc": {1: "Right ventricle", 2: "Myocardium", 3: "Left ventricle"},
    "synapse": {1: "Aorta", 2: "Gallbladder", 3: "Left kidney", 4: "Right kidney",
                5: "Liver", 6: "Pancreas", 7: "Spleen", 8: "Stomach"},
}


def patient_id(path):
    match = re.match(r"(patient\d+|case_?\d+)", Path(path).name)
    if not match:
        raise ValueError(f"Cannot identify patient: {path}")
    return match.group(1)


def read_pair(path):
    if isinstance(path, VolumeRecord):
        return path.image, path.label
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            image_key = "image" if "image" in data else "img"
            return data[image_key].copy(), data["label"].copy()
    with h5py.File(path, "r") as data:
        return data["image"][()], data["label"][()]


def validate_pair(image, label, classes):
    if image.shape != label.shape or image.ndim not in (2, 3):
        raise ValueError(f"Invalid image/mask shape: {image.shape}, {label.shape}")
    if not np.isfinite(image).all() or not np.isfinite(label).all():
        raise ValueError("Nonfinite image or mask")
    if not np.equal(label, np.floor(label)).all() or label.min() < 0 or label.max() >= classes:
        raise ValueError("Invalid integer label IDs")


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def make_split(root, dataset, destination, seed=2026):
    """Validation comes exclusively from the training pool; never infer pathology."""
    root, destination = Path(root), Path(destination)
    spec = SPECS[dataset]
    train_files = sorted((root / spec["volumes"]).glob("*.h5"))
    test_files = sorted((root / spec["test"]).glob("*.h5"))
    patients = sorted({patient_id(p) for p in train_files})
    test_ids = sorted({patient_id(p) for p in test_files})
    if not patients or not test_ids or set(patients) & set(test_ids):
        raise ValueError("Missing data or training/test patient overlap")
    validation = sorted(random.Random(seed).sample(patients, spec["val_count"]))
    training = sorted(set(patients) - set(validation))
    split = {"dataset": dataset, "seed": seed, "train": training, "val": validation,
             "test": test_ids, "protocol": "patient_holdout_from_training_pool",
             "note": "Patient-level holdout from the supplied training pool."}
    if destination.exists():
        if json.loads(destination.read_text()) != split:
            raise ValueError(f"Existing split differs; refusing overwrite: {destination}")
    else:
        write_new_json(destination, split)
    return split


@dataclass
class VolumeRecord:
    """An in-memory volume assembled from the supplied ACDC validation slices."""

    path: Path
    image: np.ndarray
    label: np.ndarray
    patient: str


def _directory(root, config, key, fallback):
    return Path(root) / config.get("data_dirs", {}).get(key, fallback)


def _acdc_validation_volumes(root, split, config, classes):
    folder = _directory(root, config, "val_slices", "valid")
    filenames = split.get("val_slice_files")
    if not filenames:
        raise ValueError("ACDC slice validation requires exact val_slice_files")
    available = {p.name: p for p in folder.glob("*.npz")}
    missing = set(filenames) - set(available)
    if missing:
        raise ValueError(f"ACDC validation manifest contains missing slice files: {sorted(missing)[:10]}")
    groups = {}
    pattern = re.compile(r"^(case_?\d+)_slice(ED|ES)_(\d+)\.npz$")
    for name in filenames:
        match = pattern.match(name)
        if not match:
            raise ValueError(f"Unrecognized ACDC validation slice name: {name}")
        patient, phase, index = match.group(1), match.group(2), int(match.group(3))
        groups.setdefault((patient, phase), []).append((index, available[name]))
    records = []
    for (patient, phase), entries in sorted(groups.items()):
        entries.sort(key=lambda item: item[0])
        indices = [index for index, _ in entries]
        if indices != list(range(len(indices))):
            raise ValueError(f"Non-contiguous ACDC validation slices: {patient} {phase}")
        pairs = [read_pair(path) for _, path in entries]
        image = np.stack([pair[0] for pair in pairs])
        label = np.stack([pair[1] for pair in pairs])
        validate_pair(image, label, classes)
        records.append(VolumeRecord(Path(f"{patient}_volume_{phase}.npz"), image, label, patient))
    if {record.patient for record in records} != set(split["val"]):
        raise ValueError("ACDC validation slices do not resolve to the exact validation patients")
    return records


def _synapse_validation_volumes(root, split, config, classes):
    """Assemble held-out Synapse training patients from the supplied NPZ slices."""
    folder = _directory(root, config, "train_slices", "Synapse/train_npz")
    selected = set(split["val"])
    pattern = re.compile(r"^(case_?\d+)_slice(\d+)\.npz$")
    groups = {}
    for path in sorted(folder.glob("*.npz")):
        match = pattern.match(path.name)
        if not match:
            continue
        patient, index = match.group(1), int(match.group(2))
        if patient in selected:
            groups.setdefault(patient, []).append((index, path))
    if set(groups) != selected:
        raise ValueError(f"Synapse validation slices do not resolve to exact patients: {sorted(set(groups) ^ selected)}")
    records = []
    for patient in sorted(selected):
        entries = sorted(groups[patient])
        indices = [index for index, _ in entries]
        if indices != list(range(len(indices))):
            raise ValueError(f"Non-contiguous Synapse validation slices: {patient}")
        pairs = [read_pair(path) for _, path in entries]
        image = np.stack([pair[0] for pair in pairs])
        label = np.stack([pair[1] for pair in pairs])
        validate_pair(image, label, classes)
        records.append(VolumeRecord(Path(f"{patient}_internal_val.npz"), image, label, patient))
    return records


def volume_paths(root, dataset, split, partition, config=None):
    if partition not in ("val", "test"):
        raise ValueError("Volume evaluation partition must be val or test")
    config = config or {}
    spec = SPECS[dataset]
    use_provided_test = partition == "test" or split.get("validation_role") == "provided_test_used_as_validation"
    if dataset == "acdc" and partition == "val" and not use_provided_test and split.get("val_slice_files"):
        return _acdc_validation_volumes(root, split, config, spec["classes"])
    if dataset == "synapse" and partition == "val" and not use_provided_test:
        return _synapse_validation_volumes(root, split, config, spec["classes"])
    key = "test_volumes" if use_provided_test else "val_volumes"
    folder = _directory(root, config, key, spec["test"] if use_provided_test else spec["volumes"])
    extensions = config.get("volume_extensions", [".h5"])
    paths = [p for p in sorted(folder.iterdir()) if p.is_file() and
             (p.suffix in extensions or any(p.name.endswith(ext) for ext in extensions)) and
             patient_id(p) in set(split[partition])]
    listed_names = split.get(f"{partition}_volume_files")
    if listed_names is not None:
        paths = [p for p in paths if p.name in set(listed_names)]
        if {p.name for p in paths} != set(listed_names):
            raise ValueError(f"Partition manifest contains missing {partition} volumes")
    if {patient_id(p) for p in paths} != set(split[partition]):
        raise ValueError("Split contains missing volume patients")
    return paths


class SliceDataset(Dataset):
    """Natural sampling, retaining every valid selected-patient slice.

    Optional RAM cache holds native or resized arrays according to the
    preprocessing profile. It writes nothing to MEDDATA and keeps input
    normalization unchanged.
    """

    def __init__(self, root, dataset, patients, size=256, augment=True, cache=True, files=None,
                 config=None):
        self.spec, self.size, self.augment = SPECS[dataset], size, augment
        self.config = config or {}
        self.resize_after_augmentation = bool(self.config.get("resize_after_augmentation", False))
        selected = set(patients)
        folder = _directory(root, self.config, "train_slices", self.spec["slices"])
        extensions = self.config.get("slice_extensions", [self.spec["extension"]])
        accepted = set(files) if files is not None else None
        self.paths = [p for p in sorted(folder.iterdir()) if p.is_file() and
                      (p.suffix in extensions) and patient_id(p) in selected and
                      (accepted is None or p.name in accepted)]
        if accepted is not None and {p.name for p in self.paths} != accepted:
            raise ValueError("Provided training list contains missing or incompatible files")
        if {patient_id(p) for p in self.paths} != selected:
            raise ValueError("Split contains missing slice patients")
        self.cached = None
        if cache:
            self.cached = [self.load(path) for path in self.paths]

    def load(self, path):
        image, label = read_pair(path)
        validate_pair(image, label, self.spec["classes"])
        if image.ndim != 2:
            raise ValueError(f"Expected 2D slice: {path}")
        image = image.astype(np.float32, copy=False)
        label = label.astype(np.uint8, copy=False)
        if self.config.get("input_scale") == "zero_one_to_minus_one_one":
            if image.min() < -1e-6 or image.max() > 1.0 + 1e-6:
                raise ValueError(f"Expected [0,1] image before configured scaling: {path}")
            image = image * 2.0 - 1.0
        elif self.config.get("input_scale") not in (None, "identity"):
            raise ValueError(f"Unsupported input_scale: {self.config['input_scale']}")
        if self.resize_after_augmentation:
            # Cache the immutable native grid; geometric augmentation must happen
            # before the final model-input resize.
            return image, label
        return self.resize_pair(image, label)

    def resize_pair(self, image, label):
        interpolation = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}.get(
            self.config.get("image_interpolation", "linear"))
        if interpolation is None:
            raise ValueError("image_interpolation must be 'linear' or 'cubic'")
        image = cv2.resize(np.ascontiguousarray(image, dtype=np.float32),
                           (self.size, self.size), interpolation=interpolation)
        label = cv2.resize(np.ascontiguousarray(label, dtype=np.uint8),
                           (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return image, label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image, label = self.cached[index] if self.cached is not None else self.load(self.paths[index])
        if self.augment:
            if self.config.get("augmentation_mode") == "flip_rotate":
                if random.random() > 0.5:
                    k = np.random.randint(0, 4)
                    image, label = np.rot90(image, k), np.rot90(label, k)
                    axis = np.random.randint(0, 2)
                    image, label = np.flip(image, axis=axis), np.flip(label, axis=axis)
                elif random.random() > 0.5:
                    from scipy import ndimage
                    angle = np.random.randint(-20, 20)
                    image_order = int(self.config.get("rotation_image_order", 0))
                    if image_order not in (0, 1, 3):
                        raise ValueError("rotation_image_order must be 0, 1, or 3")
                    image = ndimage.rotate(image, angle, order=image_order, reshape=False)
                    label = ndimage.rotate(label, angle, order=0, reshape=False)
            else:
                k = random.randrange(4)
                image, label = np.rot90(image, k), np.rot90(label, k)
                if random.random() < 0.5:
                    image, label = np.flip(image, 1), np.flip(label, 1)
                if random.random() < 0.25:
                    angle = random.uniform(-12, 12)
                    matrix = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), angle, 1.0)
                    image = cv2.warpAffine(image, matrix, (self.size, self.size), flags=cv2.INTER_LINEAR)
                    label = cv2.warpAffine(label, matrix, (self.size, self.size), flags=cv2.INTER_NEAREST)
                if random.random() < 0.3:
                    low, high = self.config.get("image_range", [0.0, 1.0])
                    image = np.clip(image * random.uniform(0.9, 1.1), low, high)
        if self.resize_after_augmentation:
            image, label = self.resize_pair(image, label)
        return (torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).unsqueeze(0),
                torch.from_numpy(np.array(label, dtype=np.int64, copy=True)))
