"""Build the model components used by UBP."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


MODEL_DEFAULTS = {
    "acdc": {"num_classes": 4, "n_gabor_filters": 5, "n_log_filters": 5},
    "synapse": {"num_classes": 9, "n_gabor_filters": 2, "n_log_filters": 5},
}


def _model_root(path: str | Path | None = None) -> Path:
    root = (Path(path) if path is not None else
            Path(__file__).resolve().parents[2] / "model_components")
    root = root.expanduser().resolve()
    if not (root / "networks" / "GLoGNet.py").is_file():
        raise FileNotFoundError(f"Model components not found under {root}")
    return root


def _model_imports(root: Path):
    root_string = str(root)
    existing = sys.modules.get("networks")
    existing_file = getattr(existing, "__file__", None)
    if existing_file and not Path(existing_file).resolve().is_relative_to(root):
        raise ImportError(
            "A different top-level 'networks' package is already imported; "
            "load UBP in an isolated process."
        )
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    wrapper_module = importlib.import_module("networks.GLoGNet")
    model_module = importlib.import_module("networks.CSUnet.vision_transformer")
    config_module = importlib.import_module("networks.CSUnet.config")
    imported = Path(wrapper_module.__file__).resolve()
    if not imported.is_relative_to(root):
        raise ImportError(f"Resolved model components outside requested source tree: {imported}")
    return wrapper_module.GLoGNet, model_module.CS_Unet, config_module.get_config


def build_backbone(
    dataset: str,
    *,
    model_root: str | Path | None = None,
    image_size: int = 224,
    with_gabor: bool = True,
    with_log: bool = True,
    n_gabor_filters: int | None = None,
    n_log_filters: int | None = None,
    gabor_kernel_size: int = 3,
    log_kernel_size: int = 3,
):
    """Build the encoder-decoder used by the saved UBP experiments.

    The returned module is the source implementation used by UBP.
    Training, device choice, data loading, logging, and checkpointing stay in
    the project harness and are deliberately not triggered here.
    """
    key = dataset.lower()
    if key not in MODEL_DEFAULTS:
        raise ValueError(f"dataset must be one of {sorted(MODEL_DEFAULTS)}")
    if image_size < 28 or image_size % 28:
        raise ValueError("image_size must be a positive multiple of 28 for the configured Swin windows")
    defaults = MODEL_DEFAULTS[key]
    n_gabor_filters = defaults["n_gabor_filters"] if n_gabor_filters is None else n_gabor_filters
    n_log_filters = defaults["n_log_filters"] if n_log_filters is None else n_log_filters
    if with_gabor and n_gabor_filters < 1:
        raise ValueError("n_gabor_filters must be positive when Gabor filters are enabled")
    if with_log and n_log_filters < 1:
        raise ValueError("n_log_filters must be positive when LoG filters are enabled")
    if gabor_kernel_size < 1 or log_kernel_size < 1:
        raise ValueError("filter kernel sizes must be positive")

    root = _model_root(model_root)
    GLoGNet, CS_Unet, get_config = _model_imports(root)
    cfg = root / "pretrained_ckpt" / "swin_tiny_patch4_window7_224_lite.yaml"
    if not cfg.is_file():
        raise FileNotFoundError(f"Expected backbone model config is missing: {cfg}")
    args = SimpleNamespace(
        cfg=str(cfg),
        opts=None,
        batch_size=1,
        zip=False,
        cache_mode="no",
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level="",
        tag=None,
        eval=False,
        throughput=False,
        with_gabor=with_gabor,
        n_gabor_filter=n_gabor_filters,
        with_LoG=with_log,
        n_LoG_filter=n_log_filters,
    )
    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = image_size
    config.MODEL.PRETRAIN_CKPT = None
    config.freeze()

    base_model = CS_Unet(config, img_size=image_size, num_classes=defaults["num_classes"])
    return GLoGNet(
        base_model,
        with_gabor=with_gabor,
        n_gabor_filters=n_gabor_filters,
        gabor_filter_kernel_size=gabor_kernel_size,
        with_LoG=with_log,
        n_LoG_filters=n_log_filters,
        LoG_filter_kernel_size=log_kernel_size,
    )
