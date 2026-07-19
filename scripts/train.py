#!/usr/bin/env python3
"""
train.py — No-annotation lung-CT classifier with a frozen 3D foundation encoder,
an evidential (Dirichlet) head, and FixMatch-style semi-supervised learning.

Pipeline (see CLAUDE.md):
  1. Load a 3D foundation encoder from ./weights (Models Genesis UNet3D by default;
     a MONAI 3D ResNet is offered as a dependency-free fallback). If no checkpoint
     is found the encoder is randomly initialized and a LOUD warning is logged so
     the run still completes unattended.
  2. Freeze most of the encoder; only the last `--unfreeze-blocks` stage(s) and the
     two heads train. Heads: a softmax classification head (cross-entropy) and an
     evidential Dirichlet head (Sensoy et al. 2018) that yields calibrated
     probabilities + an explicit uncertainty u = K / S.
  3. Read ./data/processed/manifest.csv. img_cancer (label=1) and
     img_nodule_negative (label=0) are the labeled anchors; img_nodule (label=-1,
     ~1400) is an UNLABELED pool used for FixMatch consistency / pseudo-labeling
     (weak-aug prediction supervises strong-aug prediction above a confidence gate).
  4. Evaluate on a held-out labeled test split: AUC, ECE (calibration), and
     split-conformal coverage are written to ./results/run_<ts>.json, with ROC and
     reliability plots to ./results/*.png.
  5. Fully unattended: fixed seeds, no prompts, resumable from the latest checkpoint,
     logs to ./logs/train_<ts>.log. Hard stops via --max-iterations and --max-minutes
     plus early stopping on validation AUC — it cannot run forever.

De-identification: only de-identified arrays are touched here; the manifest carries
anon_ids, never original case identifiers. Nothing is written under the read-only
source tree.

Examples:
  python scripts/train.py                               # full unattended run
  python scripts/train.py --max-iterations 50 --eval-every 25   # quick smoke test
  python scripts/train.py --backbone resnet3d --img-size 96
  python scripts/train.py --resume                      # continue latest checkpoint
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

# Headless plotting — must be set before pyplot is imported anywhere.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# --- Paths -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"
CKPT_DIR = LOG_DIR / "checkpoints"

# Filenames we recognize as published foundation checkpoints in ./weights.
GENESIS_WEIGHT_NAMES = ("Genesis_Chest_CT.pt", "Genesis_Chest_CT.pth",
                        "genesis_chest_ct.pt", "models_genesis.pt")
RESNET_WEIGHT_NAMES = ("resnet18_ct.pth", "MedicalNet_resnet18.pth",
                       "ctfm_resnet.pt")

log = logging.getLogger("train")


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# cuDNN health probe
# =============================================================================
#
# Some GPU / cuDNN combinations (observed: RTX 5090 / Blackwell sm_120 with
# cuDNN 9.x, especially under concurrent GPU load) intermittently fail 3D
# conv/batch-norm kernels with CUDNN_STATUS_EXECUTION_FAILED. Such a failure is a
# sticky CUDA error that poisons the whole process, so we cannot catch-and-retry
# in-place. Instead we probe cuDNN in an *isolated subprocess*; if it fails we
# disable cuDNN and fall back to native kernels (slower but reliable). When cuDNN
# is healthy (e.g. once preprocessing stops contending for the GPU) it stays on.

def _gpu_selftest() -> int:
    """Tiny 3D conv+BN fwd/bwd with cuDNN on. Returns 0 on success, 1 on failure.
    Invoked as `train.py --selftest` in a throwaway subprocess."""
    try:
        m = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32),
        ).cuda()
        x = torch.randn(2, 1, 48, 48, 48, device="cuda", requires_grad=True)
        with torch.autocast("cuda", enabled=True):
            y = m(x).float().mean()
        y.backward()
        torch.cuda.synchronize()
        return 0
    except Exception:  # noqa: BLE001 — any failure means "cuDNN not reliable here"
        return 1


def cudnn_is_healthy(timeout: float = 120.0) -> bool:
    import subprocess
    try:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--selftest"],
                           capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:  # noqa: BLE001 — treat probe crash/timeout as unhealthy
        return False


def configure_cudnn(mode: str, device: torch.device) -> None:
    """Set torch.backends.cudnn.{enabled,benchmark} per --cudnn {auto,on,off}."""
    if device.type != "cuda":
        return
    if mode == "on":
        torch.backends.cudnn.enabled = True
    elif mode == "off":
        torch.backends.cudnn.enabled = False
    else:  # auto
        healthy = cudnn_is_healthy()
        torch.backends.cudnn.enabled = healthy
        if not healthy:
            log.warning("cuDNN self-test FAILED — its 3D kernels are unreliable on "
                        "this GPU right now (Blackwell/cuDNN quirk, often under "
                        "concurrent GPU load). Disabling cuDNN; using native kernels "
                        "(slower but correct). Re-run with --cudnn on to force it.")
    torch.backends.cudnn.benchmark = torch.backends.cudnn.enabled
    log.info("cudnn.enabled=%s benchmark=%s", torch.backends.cudnn.enabled,
             torch.backends.cudnn.benchmark)


# =============================================================================
# Foundation encoders
# =============================================================================
#
# Both encoders expose forward(x) -> (B, feat_dim) pooled feature vector and a
# `staged_modules()` list (shallow->deep) so the freezing logic can unfreeze the
# last K stages generically.

class _ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    """BatchNorm3d that always normalizes with batch stats (Models Genesis)."""

    def _check_input_dim(self, x):
        if x.dim() != 5:
            raise ValueError(f"expected 5D input (got {x.dim()}D)")

    def forward(self, x):
        self._check_input_dim(x)
        return F.batch_norm(x, self.running_mean, self.running_var,
                            self.weight, self.bias, True, self.momentum, self.eps)


class _LUConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = _ContBatchNorm3d(out_ch)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn1(self.conv1(x)))


def _make_n_conv(in_ch, depth):
    # Mirrors Models Genesis _make_nConv so pretrained keys line up 1:1.
    layer1 = _LUConv(in_ch, 32 * (2 ** depth))
    layer2 = _LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2)
    return nn.Sequential(layer1, layer2)


class _DownTransition(nn.Module):
    def __init__(self, in_ch, depth):
        super().__init__()
        self.ops = _make_n_conv(in_ch, depth)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:           # bottleneck: no pooling
            out = self.ops(x)
        else:
            out = self.maxpool(self.ops(x))
        return out


class GenesisEncoder(nn.Module):
    """Models Genesis UNet3D *encoder* path; global-avg-pools the 512-ch bottleneck.

    Child names (down_tr64/128/256/512) match the published Genesis_Chest_CT.pt so
    its encoder weights load by simple key filtering.
    """

    feat_dim = 512

    def __init__(self):
        super().__init__()
        self.down_tr64 = _DownTransition(1, 0)
        self.down_tr128 = _DownTransition(64, 1)
        self.down_tr256 = _DownTransition(128, 2)
        self.down_tr512 = _DownTransition(256, 3)

    def forward(self, x):
        x = self.down_tr64(x)
        x = self.down_tr128(x)
        x = self.down_tr256(x)
        x = self.down_tr512(x)
        return F.adaptive_avg_pool3d(x, 1).flatten(1)   # (B, 512)

    def staged_modules(self):
        return [self.down_tr64, self.down_tr128, self.down_tr256, self.down_tr512]

    def load_foundation(self, ckpt_path: Path) -> int:
        state = _read_state_dict(ckpt_path)
        own = self.state_dict()
        matched = {}
        for k, v in state.items():
            kk = k.replace("module.", "")
            if kk in own and own[kk].shape == v.shape:
                matched[kk] = v
        self.load_state_dict(matched, strict=False)
        return len(matched)


class ResNet3DEncoder(nn.Module):
    """MONAI 3D ResNet-18 trunk (CT-FM / MedicalNet style), fc removed -> 512-d."""

    feat_dim = 512

    def __init__(self):
        super().__init__()
        from monai.networks.nets import resnet18
        net = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
        net.fc = nn.Identity()                 # forward already global-pools + flattens
        self.net = net

    def forward(self, x):
        return self.net(x)                     # (B, 512)

    def staged_modules(self):
        n = self.net
        # shallow -> deep; stem first so "unfreeze last K" peels from layer4 inward.
        return [nn.ModuleList([n.conv1, n.bn1]), n.layer1, n.layer2, n.layer3, n.layer4]

    def load_foundation(self, ckpt_path: Path) -> int:
        state = _read_state_dict(ckpt_path)
        own = self.net.state_dict()
        matched = {}
        for k, v in state.items():
            kk = k.replace("module.", "").replace("net.", "")
            if kk in own and own[kk].shape == v.shape:
                matched[kk] = v
        self.net.load_state_dict(matched, strict=False)
        return len(matched)


class MonaiModelEncoder(nn.Module):
    """Wrap MONAI classification model to extract pooled logits as features."""

    def __init__(self, model: nn.Module, out_channels: int = 2, model_type: str = "dynunet"):
        super().__init__()
        self.model = model
        self.out_channels = out_channels

    def forward(self, x):
        # MONAI models output (B, 2, H, W, D) for 2-class classification
        out = self.model(x)
        # Global average pool to (B, 2)
        if out.dim() > 2:
            out = F.adaptive_avg_pool3d(out, 1).flatten(1)
        return out

    def staged_modules(self):
        return [self.model]

    def load_foundation(self, ckpt_path: Path) -> int:
        return 0


def _read_state_dict(ckpt_path: Path) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _find_weight(names) -> Path | None:
    for n in names:
        p = WEIGHTS_DIR / n
        if p.is_file():
            return p
    return None


def build_encoder(backbone: str, weights: Path | None):
    """Construct the encoder and load foundation weights if available."""
    if backbone == "genesis_unet":
        enc = GenesisEncoder()
        ckpt = weights or _find_weight(GENESIS_WEIGHT_NAMES)
    elif backbone == "resnet3d":
        enc = ResNet3DEncoder()
        ckpt = weights or _find_weight(RESNET_WEIGHT_NAMES)
    elif backbone == "dynunet":
        from monai.networks.nets import DynUNet
        # DynUNet as encoder + projection to feat_dim
        base_model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2, 2],
            filters=[32, 64, 128, 256, 512, 1024],
            dropout=0.0,
            norm_name="instance"
        )
        # Add projection: model outputs (B,2) -> project to feat_dim
        enc = nn.Sequential(
            MonaiModelEncoder(base_model, out_channels=2, model_type="dynunet"),
            nn.Linear(2, 512)
        )
        enc.feat_dim = 512
        enc.staged_modules = lambda: [base_model]
        enc.load_foundation = lambda p: 0
        ckpt = None
    elif backbone == "unetr":
        from monai.networks.nets import UNETR
        base_model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type='conv',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            spatial_dims=3
        )
        enc = nn.Sequential(
            MonaiModelEncoder(base_model, out_channels=2, model_type="unetr"),
            nn.Linear(2, 768)
        )
        enc.feat_dim = 768
        enc.staged_modules = lambda: [base_model]
        enc.load_foundation = lambda p: 0
        ckpt = None
    elif backbone == "swinunetr":
        from monai.networks.nets import SwinUNETR
        base_model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=2,
            feature_size=24,
            use_checkpoint=False,
        )
        enc = nn.Sequential(
            MonaiModelEncoder(base_model, out_channels=2, model_type="swinunetr"),
            nn.Linear(2, 768)
        )
        enc.feat_dim = 768
        enc.staged_modules = lambda: [base_model]
        enc.load_foundation = lambda p: 0
        ckpt = None
    else:
        raise ValueError(f"unknown backbone: {backbone}")

    if ckpt is not None and Path(ckpt).is_file():
        n = enc.load_foundation(Path(ckpt))
        log.info("loaded %d pretrained tensors from %s into %s", n, ckpt, backbone)
        if n == 0:
            log.warning("checkpoint %s matched 0 tensors — architecture mismatch?", ckpt)
    else:
        log.warning("*** NO foundation checkpoint found in %s for backbone '%s'. "
                    "Encoder is RANDOMLY INITIALIZED — metrics are not meaningful "
                    "until you drop e.g. %s into ./weights. Continuing so the run "
                    "stays unattended. ***",
                    WEIGHTS_DIR, backbone, GENESIS_WEIGHT_NAMES[0])
    return enc


def freeze_encoder(encoder: nn.Module, unfreeze_last_k: int) -> None:
    """Freeze the whole encoder, then re-enable grad on its last K stages."""
    for p in encoder.parameters():
        p.requires_grad = False
    stages = encoder.staged_modules()
    for stage in stages[len(stages) - max(0, unfreeze_last_k):]:
        for p in stage.parameters():
            p.requires_grad = True


def set_train_mode(model: nn.Module, encoder: nn.Module, unfreeze_last_k: int) -> None:
    """Put the model in train mode, but keep BatchNorm of the *frozen* encoder
    stages in eval mode — a frozen pretrained backbone should use its running
    statistics, and this also avoids BN-training kernels on the frozen majority."""
    model.train()
    stages = encoder.staged_modules()
    for stage in stages[:len(stages) - max(0, unfreeze_last_k)]:
        stage.eval()


# =============================================================================
# Model: encoder + classification head + evidential (Dirichlet) head
# =============================================================================

class CTClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_classes: int = 2, p_drop: float = 0.3):
        super().__init__()
        self.encoder = encoder
        d = encoder.feat_dim
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p_drop)
        self.cls_head = nn.Linear(d, n_classes)   # softmax / cross-entropy
        self.evi_head = nn.Linear(d, n_classes)   # Dirichlet evidence (via softplus)

    def forward(self, x):
        f = self.drop(self.norm(self.encoder(x)))
        return self.cls_head(f), self.evi_head(f)


# --- Evidential (Dirichlet) machinery ----------------------------------------

def evidence_to_alpha(evi_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(evi_logits) + 1.0            # alpha_k = e_k + 1 >= 1


def dirichlet_prob_unc(alpha: torch.Tensor):
    """Expected categorical prob p = alpha/S and vacuity uncertainty u = K/S."""
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    u = alpha.shape[1] / S.squeeze(1)
    return p, u


def _kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1,...,1) ), per-sample."""
    K = alpha.shape[1]
    ones = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    first = (torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
             - torch.lgamma(ones.sum(dim=1, keepdim=True)))
    dg = torch.digamma(alpha) - torch.digamma(sum_alpha)
    second = ((alpha - ones) * dg).sum(dim=1, keepdim=True)
    return (first + second).squeeze(1)


def edl_mse_loss(evi_logits, targets, n_classes, anneal_coef):
    """Bayes-risk MSE evidential loss + annealed KL regularizer (Sensoy 2018)."""
    alpha = evidence_to_alpha(evi_logits)
    y = F.one_hot(targets, n_classes).float()
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    err = ((y - p) ** 2).sum(dim=1)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1)
    alpha_tilde = y + (1.0 - y) * alpha          # strip evidence from the true class
    kl = _kl_dirichlet_uniform(alpha_tilde)
    return (err + var + anneal_coef * kl).mean()


# =============================================================================
# Data
# =============================================================================

@dataclass
class Sample:
    path: Path
    label: int        # 1 / 0 / -1 (unlabeled)
    role: str
    anon_id: str


def read_manifest(proc_dir: Path) -> list[Sample]:
    """Parse manifest.csv -> de-duplicated, existing-on-disk samples.

    The manifest may contain `skipped_exists` rows and repeats; we keep one row
    per anon_id with an 'ok' status and a present .npz file. Robust to a manifest
    that is still being appended to by a concurrent preprocess run.
    """
    mpath = proc_dir / "manifest.csv"
    if not mpath.is_file():
        raise FileNotFoundError(f"manifest not found: {mpath}")
    seen: dict[str, Sample] = {}
    with open(mpath, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in ("ok", "ok_no_lung"):
                continue
            aid = row["anon_id"]
            if aid in seen:
                continue
            npz = proc_dir / row["class_dir"] / f"{aid}.npz"
            if not npz.is_file():
                continue
            try:
                label = int(row["label"])
            except (TypeError, ValueError):
                continue
            seen[aid] = Sample(npz, label, row.get("role", ""), aid)
    return list(seen.values())


def stratified_split(samples, fracs, seed):
    """Stratified train/cal/test split of labeled samples. Degrades gracefully
    when a class is tiny (falls back to a plain shuffled split)."""
    f_train, f_cal, f_test = fracs
    rng = np.random.default_rng(seed)
    by_cls = {0: [], 1: []}
    for s in samples:
        by_cls[s.label].append(s)
    train, cal, test = [], [], []
    for cls, items in by_cls.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * f_test))
        n_cal = int(round(n * f_cal))
        # Guarantee at least one test/cal item per class when the class is non-empty.
        if n >= 3:
            n_test = max(1, n_test)
            n_cal = max(1, n_cal)
        n_cal = min(n_cal, max(0, n - n_test - 1)) if n >= 3 else n_cal
        test += items[:n_test]
        cal += items[n_test:n_test + n_cal]
        train += items[n_test + n_cal:]
    rng.shuffle(train); rng.shuffle(cal); rng.shuffle(test)
    return train, cal, test


def _patient_groups(samples):
    """Patient IDs for group-aware CV (same logic as run_saturation.py: id_map + strip _B\\d/_\\d)."""
    import csv, re
    idmap = {r["anon_id"]: r["original_stem"]
             for r in csv.DictReader(open(PROC_DIR / "id_map.csv"))}
    pat = lambda a: re.sub(r"_B\d+$|_\d+$", "", idmap.get(a, a))
    return np.array([pat(s.anon_id) for s in samples])


def grouped_split(samples, n_folds, fold, seed, cal_frac=0.2):
    """Patient-grouped stratified CV: fold=test; the rest is split (also grouped) into train/cal.
    No patient appears in more than one of train/cal/test (prevents same-patient leakage)."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.array([s.label for s in samples]); g = _patient_groups(samples)
    outer = list(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(samples, y, g))
    tr_idx, te_idx = outer[fold]
    tr = [samples[i] for i in tr_idx]; test = [samples[i] for i in te_idx]
    k = max(2, int(round(1.0 / cal_frac)))
    inner = list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(tr, y[tr_idx], g[tr_idx]))[0]
    t2, c2 = inner
    return [tr[i] for i in t2], [tr[i] for i in c2], test


def build_transforms(img_size: int, hu_min: int, hu_max: int):
    """Return (eval_tf, weak_tf, strong_tf) operating on channel-first (1,D,H,W)."""
    from monai.transforms import (
        Compose, ScaleIntensityRange, Resize, RandFlip, RandAffine,
        RandGaussianNoise, RandShiftIntensity, RandAdjustContrast,
        RandCoarseDropout, EnsureType,
    )
    size = (img_size, img_size, img_size)
    base = [
        ScaleIntensityRange(a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
        Resize(spatial_size=size, mode="trilinear", align_corners=False),
    ]
    to_t = [EnsureType(data_type="tensor", dtype=torch.float32)]
    eval_tf = Compose(base + to_t)
    weak_tf = Compose(base + [
        RandFlip(prob=0.5, spatial_axis=0),
        RandAffine(prob=0.5, rotate_range=(0.05, 0.05, 0.05),
                   scale_range=(0.05, 0.05, 0.05), mode="bilinear", padding_mode="zeros"),
    ] + to_t)
    strong_tf = Compose(base + [
        RandFlip(prob=0.5, spatial_axis=0),
        RandAffine(prob=0.8, rotate_range=(0.2, 0.2, 0.2),
                   translate_range=(8, 8, 8), scale_range=(0.15, 0.15, 0.15),
                   mode="bilinear", padding_mode="zeros"),
        RandGaussianNoise(prob=0.5, std=0.05),
        RandShiftIntensity(prob=0.5, offsets=0.1),
        RandAdjustContrast(prob=0.5, gamma=(0.7, 1.5)),
        RandCoarseDropout(prob=0.5, holes=4, spatial_size=img_size // 8,
                          max_spatial_size=img_size // 4, fill_value=0.0),
    ] + to_t)
    return eval_tf, weak_tf, strong_tf


def _load_volume(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as d:
        img = d["image"].astype(np.float32)        # (D,H,W) HU
    return img[None]                                # add channel -> (1,D,H,W)


class LabeledDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = self.transform(_load_volume(s.path))
        return torch.as_tensor(np.asarray(x), dtype=torch.float32), s.label


class UnlabeledDataset(Dataset):
    """Returns a (weak, strong) augmented pair of the same volume for FixMatch."""

    def __init__(self, samples, weak_tf, strong_tf):
        self.samples = samples
        self.weak_tf = weak_tf
        self.strong_tf = strong_tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vol = _load_volume(self.samples[i].path)
        w = torch.as_tensor(np.asarray(self.weak_tf(vol)), dtype=torch.float32)
        s = torch.as_tensor(np.asarray(self.strong_tf(vol)), dtype=torch.float32)
        return w, s


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


# =============================================================================
# Metrics
# =============================================================================

def expected_calibration_error(probs_pos, labels, n_bins=10):
    probs_pos = np.asarray(probs_pos)
    labels = np.asarray(labels)
    conf = np.where(probs_pos >= 0.5, probs_pos, 1 - probs_pos)
    pred = (probs_pos >= 0.5).astype(int)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def conformal_lac(cal_probs, cal_labels, test_probs, test_labels, alpha):
    """Split-conformal LAC: nonconformity = 1 - p[true]. Returns coverage + set size.

    cal_probs/test_probs are (N, K) expected categorical probabilities.
    """
    cal_probs = np.asarray(cal_probs)
    cal_labels = np.asarray(cal_labels)
    n = len(cal_labels)
    if n == 0:
        return {"alpha": alpha, "qhat": None, "coverage": None,
                "avg_set_size": None, "n_cal": 0, "note": "no calibration data"}
    scores = 1.0 - cal_probs[np.arange(n), cal_labels]
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = float(np.quantile(scores, level, method="higher"))
    test_probs = np.asarray(test_probs)
    in_set = test_probs >= (1.0 - qhat)            # (M, K) boolean prediction sets
    sizes = in_set.sum(axis=1)
    covered = in_set[np.arange(len(test_labels)), np.asarray(test_labels)]
    return {
        "alpha": float(alpha),
        "target_coverage": float(1 - alpha),
        "qhat": qhat,
        "coverage": float(np.mean(covered)) if len(test_labels) else None,
        "avg_set_size": float(np.mean(sizes)) if len(sizes) else None,
        "n_cal": int(n),
        "n_test": int(len(test_labels)),
    }


def fit_temperature(probs, labels):
    """Fit a single softmax temperature T>0 by minimizing NLL on (probs, labels).

    Operates on the predictive categorical distribution: scaled = softmax(log(p)/T).
    T>1 softens an over-confident model. Returns 1.0 (no-op) if it can't fit
    (one class only, too few samples, or a numerical failure)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return 1.0
    logp = torch.log(torch.as_tensor(probs, dtype=torch.float64).clamp_min(1e-8))
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(log_t) > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logp / torch.exp(log_t), y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        T = float(torch.exp(log_t).item())
        return T if (math.isfinite(T) and T > 0) else 1.0
    except Exception:  # noqa: BLE001 — calibration is best-effort; fall back to T=1
        return 1.0


def apply_temperature(probs, T):
    """Temperature-scale predictive probabilities: softmax(log(p)/T), row-normalized."""
    probs = np.asarray(probs, dtype=np.float64)
    logits = np.log(np.clip(probs, 1e-8, None)) / float(T)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def safe_auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


# =============================================================================
# Evaluation pass
# =============================================================================

@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    probs, uncs, labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            _, evi = model(x)
        alpha = evidence_to_alpha(evi.float())
        p, u = dirichlet_prob_unc(alpha)
        probs.append(p.cpu().numpy())
        uncs.append(u.cpu().numpy())
        labels.append(y.numpy())
    if not probs:
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    return np.concatenate(probs), np.concatenate(uncs), np.concatenate(labels)


def plot_roc(labels, scores, out_path, title):
    from sklearn.metrics import roc_curve, roc_auc_score
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(title); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out_path, dpi=130); plt.close()
    return str(out_path)


def plot_reliability(labels, probs_pos, out_path, title, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs = [], []
    labels = np.asarray(labels); probs_pos = np.asarray(probs_pos)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs_pos > lo) & (probs_pos <= hi)
        if m.sum() == 0:
            continue
        centers.append(probs_pos[m].mean())
        accs.append(labels[m].mean())
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect")
    plt.plot(centers, accs, "o-", label="model")
    plt.xlabel("Mean predicted P(cancer)"); plt.ylabel("Empirical fraction positive")
    plt.title(title); plt.legend(loc="upper left"); plt.tight_layout()
    plt.savefig(out_path, dpi=130); plt.close()
    return str(out_path)


# =============================================================================
# Checkpointing
# =============================================================================

def save_ckpt(path: Path, model, opt, scaler, it, best, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "iteration": it,
        "best": best,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": asdict(cfg),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, tmp)
    os.replace(tmp, path)


def load_ckpt(path: Path, model, opt, scaler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    rng = ck.get("rng", {})
    try:
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            # map_location may have moved these onto the GPU; set_rng_state_all
            # requires CPU ByteTensors.
            torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s
                                          for s in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
    except Exception:                              # noqa: BLE001 — RNG restore is best-effort
        log.warning("could not fully restore RNG state from checkpoint")
    return ck.get("iteration", 0), ck.get("best", -1.0)


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    backbone: str = "genesis_unet"
    weights: str | None = None
    img_size: int = 96
    hu_min: int = -1000
    hu_max: int = 400
    unfreeze_blocks: int = 1
    dropout: float = 0.3
    # optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    labeled_bs: int = 4
    unlabeled_bs: int = 4
    num_workers: int = 1
    # semi-supervised (FixMatch / Mean Teacher)
    lambda_u: float = 1.0
    ssl: str = "fixmatch"                # fixmatch | meanteacher (pseudo-label source)
    ema_decay: float = 0.999            # Mean-Teacher EMA decay for the teacher
    gate: str = "hybrid"                 # confidence | hybrid | evidential (pseudo-label gate)
    gate_temp: float = 1.0              # softmax-confidence temperature (<1 sharpens)
    gate_type: str = "original"         # original | temp_conf | entropy (pseudo-label gating strategy)
    conf_threshold: float = 0.95
    entropy_threshold: float = 0.5       # max entropy H(p) for entropy-based gating
    unc_threshold: float = 0.5          # max vacuity u for a pseudo-label to count (hybrid/evidential)
    u_rampup: int = 400
    edl_anneal: int = 1000
    # splits
    test_frac: float = 0.2
    cal_frac: float = 0.2
    # QC-clean allowlist + patient-grouped CV + cross-site transfer (Study 1 DL)
    case_list: str = ""
    n_folds: int = 0            # 0 = stratified single split; >0 = patient-grouped CV
    cv_fold: int = 0
    cv_seed: int = 0
    eval_lidc: str = ""         # LIDC processed dir -> cross-site transfer AUC
    # stopping / schedule
    max_iterations: int = 3000
    max_minutes: float = 240.0
    max_labeled: int = 0                # 0 = use all labeled train; >0 = low-label regime
    eval_every: int = 100
    ckpt_every: int = 100
    patience: int = 10                  # early-stop after N evals w/o val-AUC gain
    warmup: int = 50
    # conformal + calibration
    conformal_alpha: float = 0.1
    temperature: bool = True     # temperature-scale probabilities before conformal
    temp_frac: float = 0.5       # fraction of the cal split used to fit T (rest -> conformal)
    # misc
    seed: int = 1337
    no_amp: bool = False
    cudnn: str = "auto"          # auto | on | off  (auto self-tests + falls back)
    resume: bool = False
    eval_only: bool = False      # skip training: load best.pt and just re-evaluate
    tag: str = ""
    ckpt_dir: str = ""           # override checkpoint dir (default: logs/checkpoints)


# =============================================================================
# Training
# =============================================================================

def train(cfg: Config, ts: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not cfg.no_amp) and device.type == "cuda"
    log.info("device=%s amp=%s", device, use_amp)
    configure_cudnn(cfg.cudnn, device)

    # --- data ----------------------------------------------------------------
    samples = read_manifest(PROC_DIR)
    labeled = [s for s in samples if s.label in (0, 1)]
    unlabeled = [s for s in samples if s.label == -1]
    n_pos = sum(s.label == 1 for s in labeled)
    n_neg = sum(s.label == 0 for s in labeled)
    log.info("samples: labeled=%d (pos=%d neg=%d) unlabeled=%d",
             len(labeled), n_pos, n_neg, len(unlabeled))
    if len(labeled) < 2:
        raise RuntimeError("need at least 2 labeled anchors to train; "
                           "is preprocessing finished?")

    if cfg.case_list:
        cl = json.load(open(cfg.case_list))["sets"]
        good = set(cl["cancer"]["good"]) | set(cl["neg"]["good"])
        before = len(labeled)
        labeled = [s for s in labeled if s.path.name in good]
        log.info("case-list %s: kept %d/%d labeled anchors", cfg.case_list, len(labeled), before)

    if cfg.n_folds > 0:
        train_l, cal_l, test_l = grouped_split(labeled, cfg.n_folds, cfg.cv_fold, cfg.cv_seed)
        log.info("patient-grouped CV: fold %d/%d (cv_seed=%d)", cfg.cv_fold, cfg.n_folds, cfg.cv_seed)
    else:
        train_l, cal_l, test_l = stratified_split(
            labeled, (1 - cfg.test_frac - cfg.cal_frac, cfg.cal_frac, cfg.test_frac), cfg.seed)

    # Optional low-label regime: subsample the labeled TRAIN set (stratified by class),
    # leaving cal/test untouched so evaluation is comparable across label budgets.
    if cfg.max_labeled and cfg.max_labeled < len(train_l):
        rng = np.random.RandomState(cfg.seed)
        pos = [s for s in train_l if s.label == 1]
        neg = [s for s in train_l if s.label == 0]
        k = cfg.max_labeled // 2
        rng.shuffle(pos); rng.shuffle(neg)
        train_l = pos[:k] + neg[:cfg.max_labeled - k]
        rng.shuffle(train_l)
        log.info("low-label regime: subsampled train to %d labeled (pos=%d neg=%d)",
                 len(train_l), sum(s.label == 1 for s in train_l),
                 sum(s.label == 0 for s in train_l))

    log.info("split: train=%d cal=%d test=%d", len(train_l), len(cal_l), len(test_l))

    eval_tf, weak_tf, strong_tf = build_transforms(cfg.img_size, cfg.hu_min, cfg.hu_max)

    # Class-balanced sampling for the labeled stream.
    train_labels = np.array([s.label for s in train_l])
    cls_count = np.bincount(train_labels, minlength=2).astype(float)
    w_per_cls = 1.0 / np.clip(cls_count, 1.0, None)
    sample_w = w_per_cls[train_labels]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                    num_samples=len(train_l), replacement=True)

    pin = device.type == "cuda"
    labeled_loader = DataLoader(
        LabeledDataset(train_l, weak_tf), batch_size=min(cfg.labeled_bs, len(train_l)),
        sampler=sampler, num_workers=cfg.num_workers, pin_memory=pin, drop_last=False)

    have_unlabeled = len(unlabeled) > 0
    if have_unlabeled:
        unlabeled_loader = DataLoader(
            UnlabeledDataset(unlabeled, weak_tf, strong_tf),
            batch_size=min(cfg.unlabeled_bs, len(unlabeled)), shuffle=True,
            num_workers=cfg.num_workers, pin_memory=pin, drop_last=True)
    else:
        log.warning("no unlabeled pool found — running supervised only (lambda_u=0).")

    def eval_loader(items):
        if not items:
            return None
        return DataLoader(LabeledDataset(items, eval_tf),
                          batch_size=max(1, cfg.labeled_bs), shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=pin)

    val_loader = eval_loader(cal_l)        # cal split doubles as validation for early-stop
    test_loader = eval_loader(test_l)
    cal_loader = eval_loader(cal_l)

    # --- model ---------------------------------------------------------------
    encoder = build_encoder(cfg.backbone, Path(cfg.weights) if cfg.weights else None)
    freeze_encoder(encoder, cfg.unfreeze_blocks)
    model = CTClassifier(encoder, n_classes=2, p_drop=cfg.dropout).to(device)
    # Mean Teacher: an EMA copy of the student that generates pseudo-labels and is the
    # inference model. Created only for training (eval-only just loads best.pt into `model`).
    teacher = None
    if cfg.ssl == "meanteacher" and not cfg.eval_only:
        import copy
        teacher = copy.deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
    eval_model = teacher if teacher is not None else model   # used for val/checkpoint/inference
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in trainable)
    n_all_p = sum(p.numel() for p in model.parameters())
    log.info("trainable params: %.2fM / %.2fM", n_train_p / 1e6, n_all_p / 1e6)

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_iterations)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- resume --------------------------------------------------------------
    ckpt_dir = Path(cfg.ckpt_dir) if cfg.ckpt_dir else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    start_it, best_auc = 0, -1.0
    if cfg.resume and ckpt_path.is_file():
        start_it, best_auc = load_ckpt(ckpt_path, model, opt, scaler, device)
        for _ in range(start_it):
            sched.step()
        log.info("resumed from iter %d (best val AUC=%.4f)", start_it, best_auc)

    if cfg.eval_only:
        # Skip training entirely: load a trained checkpoint (prefer best.pt) and
        # jump straight to evaluation — used to re-calibrate / regenerate the
        # results JSON without a re-run.
        src = best_path if best_path.is_file() else ckpt_path
        if not src.is_file():
            raise RuntimeError(f"--eval-only requires a trained checkpoint; none in {ckpt_dir}")
        start_it, best_auc = load_ckpt(src, model, opt, scaler, device)
        log.info("eval-only: loaded %s @ iter %d (best val AUC=%.4f)",
                 src.name, start_it, best_auc)

    labeled_iter = infinite(labeled_loader)
    unlabeled_iter = infinite(unlabeled_loader) if have_unlabeled else None

    history = []
    evals_without_gain = 0
    t_start = time.time()
    stop_reason = "eval_only" if cfg.eval_only else "max_iterations"

    # Accumulators for FixMatch metrics (reset per eval cycle)
    mask_rates = []
    pseudo_pos_list = []

    set_train_mode(model, encoder, cfg.unfreeze_blocks)
    it = start_it - 1                       # so final_it is right if the loop never runs
    loop_end = start_it if cfg.eval_only else cfg.max_iterations
    for it in range(start_it, loop_end):
        # ----- supervised -----
        xb, yb = next(labeled_iter)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        anneal = min(1.0, (it + 1) / max(1, cfg.edl_anneal))

        with torch.autocast("cuda", enabled=use_amp):
            cls_l, evi_l = model(xb)
            loss_ce = F.cross_entropy(cls_l, yb)
            loss_edl = edl_mse_loss(evi_l, yb, 2, anneal)
            sup_loss = loss_ce + loss_edl

            # ----- FixMatch on the unlabeled pool -----
            fix_loss = torch.zeros((), device=device)
            mask_rate = 0.0
            pseudo_pos = float("nan")
            # lambda_u==0 -> skip the unlabeled pool ENTIRELY (no forward pass), so
            # the labeled-only ablation never touches img_nodule, not even via
            # BatchNorm running-stat updates from the no-grad forward below.
            if have_unlabeled and it >= cfg.warmup and cfg.lambda_u > 0:
                xw, xs = next(unlabeled_iter)
                xw = xw.to(device, non_blocking=True)
                xs = xs.to(device, non_blocking=True)
                with torch.no_grad():
                    # FixMatch: pseudo-labels from the student itself.
                    # Mean Teacher: pseudo-labels from the EMA teacher (more stable).
                    cls_w, evi_w = (teacher if teacher is not None else model)(xw)
                    u_w = dirichlet_prob_unc(evidence_to_alpha(evi_w.float()))[1]
                    p_w = F.softmax(cls_w.float() / cfg.gate_temp, dim=1)
                    conf, pseudo = p_w.max(dim=1)

                    # Three gating options:
                    # Option 1 (original): vacuity + softmax confidence
                    if cfg.gate_type == "original":
                        keep = (conf >= cfg.conf_threshold) & (u_w <= cfg.unc_threshold)
                    # Option 2 (temp_conf, recommended): confidence alone (vacuity-free)
                    # Rationale: vacuity filter too strict post-temp-scaling; confidence sufficient
                    elif cfg.gate_type == "temp_conf":
                        keep = conf >= cfg.conf_threshold
                    # Option 3 (entropy): entropy-based gating (vacuity-free)
                    elif cfg.gate_type == "entropy":
                        entropy = -(p_w * torch.log(p_w + 1e-8)).sum(dim=1)
                        keep = entropy <= cfg.entropy_threshold
                    # Default (backward compat): hybrid (confidence + vacuity)
                    else:
                        keep = (conf >= cfg.conf_threshold) & (u_w <= cfg.unc_threshold)

                    mask = keep
                    mask_rate = float(mask.float().mean().item())
                    # confirmation-bias monitor: fraction of pseudo-labels that are class 1
                    pseudo_pos = float(pseudo[mask].float().mean().item()) if mask.any() else float("nan")
                if mask.any():
                    cls_s, evi_s = model(xs)
                    ce_s = F.cross_entropy(cls_s, pseudo, reduction="none")
                    fix_loss = (ce_s * mask.float()).sum() / mask.float().sum()

            lam = cfg.lambda_u * min(1.0, (it + 1) / max(1, cfg.u_rampup))
            loss = sup_loss + lam * fix_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        # ----- Mean-Teacher EMA update (teacher <- decay*teacher + (1-decay)*student) -----
        if teacher is not None:
            with torch.no_grad():
                d = cfg.ema_decay
                for tp, sp in zip(teacher.parameters(), model.parameters()):
                    tp.mul_(d).add_(sp.detach(), alpha=1.0 - d)
                for tb, sb in zip(teacher.buffers(), model.buffers()):
                    tb.copy_(sb)        # BN running stats tracked from the student

        # Accumulate FixMatch metrics for history
        mask_rates.append(mask_rate)
        pseudo_pos_list.append(pseudo_pos)

        if (it + 1) % 20 == 0:
            log.info("it %d/%d loss=%.4f (ce=%.3f edl=%.3f fix=%.3f) lam=%.2f mask=%.2f ppos=%.2f",
                     it + 1, cfg.max_iterations, loss.item(), loss_ce.item(),
                     loss_edl.item(), float(fix_loss.detach()), lam, mask_rate, pseudo_pos)

        # ----- periodic checkpoint -----
        if (it + 1) % cfg.ckpt_every == 0:
            save_ckpt(ckpt_path, model, opt, scaler, it + 1, best_auc, cfg)

        # ----- periodic validation + early stopping -----
        if val_loader is not None and (it + 1) % cfg.eval_every == 0:
            vp, vu, vy = predict(eval_model, val_loader, device, use_amp)
            val_auc = safe_auc(vy, vp[:, 1]) if len(vy) else None
            val_ece = expected_calibration_error(vp[:, 1], vy) if len(vy) else None
            # Compute average FixMatch metrics since last eval
            avg_mask_rate = float(np.mean(mask_rates)) if mask_rates else None
            avg_pseudo_pos = float(np.nanmean(pseudo_pos_list)) if pseudo_pos_list else None
            history.append({"iter": it + 1, "val_auc": val_auc, "val_ece": val_ece,
                            "mean_unc": float(np.mean(vu)) if len(vu) else None,
                            "avg_mask_rate": avg_mask_rate, "avg_pseudo_pos": avg_pseudo_pos})
            mask_rates = []
            pseudo_pos_list = []
            log.info("[eval] it %d val_auc=%s val_ece=%s", it + 1,
                     f"{val_auc:.4f}" if val_auc is not None else "n/a",
                     f"{val_ece:.4f}" if val_ece is not None else "n/a")
            score = val_auc if val_auc is not None else -1.0
            if score > best_auc:
                best_auc = score
                evals_without_gain = 0
                save_ckpt(best_path, eval_model, opt, scaler, it + 1, best_auc, cfg)
            else:
                evals_without_gain += 1
                if val_auc is not None and evals_without_gain >= cfg.patience:
                    stop_reason = "early_stopping"
                    log.info("early stopping: no val-AUC gain for %d evals", cfg.patience)
                    break
            set_train_mode(model, encoder, cfg.unfreeze_blocks)

        # ----- wall-clock guard -----
        if (time.time() - t_start) / 60.0 >= cfg.max_minutes:
            stop_reason = "max_minutes"
            log.info("hit max wall-clock of %.1f min — stopping", cfg.max_minutes)
            break

    final_it = it + 1
    if not cfg.eval_only:
        save_ckpt(ckpt_path, model, opt, scaler, final_it, best_auc, cfg)

    # --- final evaluation with the best checkpoint ---------------------------
    if not cfg.eval_only and best_path.is_file():
        load_ckpt(best_path, model, opt, scaler, device)
        log.info("loaded best checkpoint for final evaluation")

    results = {
        "timestamp": ts,
        "stop_reason": stop_reason,
        "iterations_run": final_it,
        "config": asdict(cfg),
        "device": str(device),
        "data": {"n_labeled": len(labeled), "n_pos": int(n_pos), "n_neg": int(n_neg),
                 "n_unlabeled": len(unlabeled), "n_train": len(train_l),
                 "n_cal": len(cal_l), "n_test": len(test_l)},
        "history": history,
        "best_val_auc": None if best_auc < 0 else best_auc,
    }

    if test_loader is not None:
        tp, tu, ty = predict(model, test_loader, device, use_amp)
        cp, cu, cy = predict(model, cal_loader, device, use_amp) if cal_loader else (
            np.empty((0, 2)), np.empty(0), np.empty(0))
        if len(ty):
            from sklearn.metrics import average_precision_score, brier_score_loss
            cy_i, ty_i = cy.astype(int), ty.astype(int)

            # --- Temperature scaling (Guo et al. 2017), fit before conformal -----
            # Split the cal set: one disjoint slice fits T, the other calibrates
            # conformal — so the conformal calibration data stays independent of the
            # temperature fit (reusing it would bias coverage). T is applied to the
            # conformal-cal slice and the test set. AUC is T-invariant (monotonic).
            T = 1.0
            if cfg.temperature and len(cy_i) >= 4 and len(np.unique(cy_i)) > 1:
                rng = np.random.default_rng(cfg.seed)
                perm = rng.permutation(len(cy_i))
                n_t = max(2, int(round(len(cy_i) * cfg.temp_frac)))
                n_t = min(n_t, len(cy_i) - 2)   # leave >=2 for conformal
                ti, ci = perm[:n_t], perm[n_t:]
                T = fit_temperature(cp[ti], cy_i[ti])
                log.info("temperature T=%.3f (fit on %d cal, conformal on %d cal)",
                         T, len(ti), len(ci))
            else:
                ci = np.arange(len(cy_i))       # no scaling -> conformal on full cal

            cp_conf_raw, cy_conf = cp[ci], cy_i[ci]
            cp_conf_cal = apply_temperature(cp_conf_raw, T)   # T=1.0 -> identity
            tp_cal = apply_temperature(tp, T)

            auc = safe_auc(ty_i, tp[:, 1])      # ranking — unaffected by T
            ece_raw = expected_calibration_error(tp[:, 1], ty_i)
            ece_cal = expected_calibration_error(tp_cal[:, 1], ty_i)
            results["metrics"] = {
                "test_auc": auc,
                "test_auprc": (float(average_precision_score(ty_i, tp[:, 1]))
                               if len(np.unique(ty_i)) > 1 else None),
                "test_ece": ece_cal,            # primary = calibrated
                "test_ece_raw": ece_raw,
                "test_brier": float(brier_score_loss(ty_i, tp_cal[:, 1]))
                              if len(np.unique(ty_i)) > 1 else None,
                "test_brier_raw": float(brier_score_loss(ty_i, tp[:, 1]))
                                  if len(np.unique(ty_i)) > 1 else None,
                "temperature": T,
                "mean_uncertainty": float(np.mean(tu)),
            }
            # Conformal on temperature-scaled probabilities (+ raw, for comparison)
            results["conformal"] = conformal_lac(
                cp_conf_cal, cy_conf, tp_cal, ty_i, cfg.conformal_alpha)
            results["conformal_raw"] = conformal_lac(
                cp_conf_raw, cy_conf, tp, ty_i, cfg.conformal_alpha)
            stem = f"run_{cfg.tag}_{ts}" if cfg.tag else f"run_{ts}"
            results["plots"] = {
                "roc": plot_roc(ty_i, tp[:, 1], RESULTS_DIR / f"{stem}_roc.png",
                                f"ROC ({cfg.backbone})"),
                "reliability": plot_reliability(ty_i, tp_cal[:, 1],
                                RESULTS_DIR / f"{stem}_calibration.png",
                                f"Calibration ({cfg.backbone}, T={T:.2f})"),
            }
            log.info("TEST auc=%s ece_raw=%.4f ece_cal=%.4f T=%.3f conformal_cov=%s",
                     f"{auc:.4f}" if auc is not None else "n/a", ece_raw, ece_cal, T,
                     results["conformal"].get("coverage"))
        else:
            results["metrics"] = {"note": "empty test split"}
    else:
        results["metrics"] = {"note": "no labeled test split available"}

    # cross-site transfer: run the best hospital-trained model on external LIDC (same encoder/tf)
    if cfg.eval_lidc:
        from collections import namedtuple as _nt
        LS = _nt("LS", ["path", "label", "anon_id"])
        lsamples = []
        for f in sorted(Path(cfg.eval_lidc).glob("*.npz")):
            with np.load(f, allow_pickle=True) as d:
                lab = int(d["label"]) if "label" in d.files else -1
                aid = str(d["anon_id"]) if "anon_id" in d.files else f.stem
            if lab in (0, 1):
                lsamples.append(LS(f, lab, aid))
        if lsamples:
            lloader = DataLoader(LabeledDataset(lsamples, eval_tf),
                                 batch_size=8, shuffle=False, num_workers=1, pin_memory=True)
            lp, lu, ly = predict(model, lloader, device, use_amp)
            lauc = safe_auc(ly, lp[:, 1])
            results["crosssite"] = {"lidc_auc": lauc, "n": int(len(ly)),
                                    "n_mal": int((ly == 1).sum()), "n_ben": int((ly == 0).sum())}
            log.info("CROSS-SITE LIDC transfer: auc=%s (n=%d mal=%d ben=%d)",
                     f"{lauc:.4f}" if lauc is not None else "n/a",
                     len(ly), int((ly == 1).sum()), int((ly == 0).sum()))

    return results


# =============================================================================
# Entry point
# =============================================================================

def parse_args(argv):
    d = Config()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=["genesis_unet", "resnet3d", "dynunet", "unetr", "swinunetr"], default=d.backbone)
    p.add_argument("--weights", default=d.weights,
                   help="explicit foundation checkpoint path (default: auto-search ./weights)")
    p.add_argument("--img-size", type=int, default=d.img_size)
    p.add_argument("--hu-min", type=int, default=d.hu_min)
    p.add_argument("--hu-max", type=int, default=d.hu_max)
    p.add_argument("--unfreeze-blocks", type=int, default=d.unfreeze_blocks,
                   help="number of trailing encoder stages to fine-tune (0 = linear probe)")
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--labeled-bs", type=int, default=d.labeled_bs)
    p.add_argument("--unlabeled-bs", type=int, default=d.unlabeled_bs)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--lambda-u", type=float, default=d.lambda_u)
    p.add_argument("--ssl", choices=["fixmatch", "meanteacher"], default=d.ssl,
                   help="pseudo-label source: fixmatch=self, meanteacher=EMA teacher")
    p.add_argument("--ema-decay", type=float, default=d.ema_decay,
                   help="Mean-Teacher EMA decay (teacher = decay*teacher + (1-decay)*student)")
    p.add_argument("--gate", choices=["confidence", "hybrid", "evidential"], default=d.gate,
                   help="pseudo-label gate: confidence=softmax(cls); hybrid=+vacuity filter; "
                        "evidential=legacy evidential-prob (inert, for the control run)")
    p.add_argument("--gate-temp", type=float, default=d.gate_temp,
                   help="softmax-confidence temperature for gating (<1 sharpens)")
    p.add_argument("--conf-threshold", type=float, default=d.conf_threshold)
    p.add_argument("--gate-type", choices=["original", "temp_conf", "entropy"], default=d.gate_type,
                   help="pseudo-label gating strategy: original=vacuity+conf; "
                        "temp_conf=temperature-scaled confidence; entropy=entropy-based")
    p.add_argument("--entropy-threshold", type=float, default=d.entropy_threshold,
                   help="max entropy H(p) for entropy-based gating")
    p.add_argument("--unc-threshold", type=float, default=d.unc_threshold)
    p.add_argument("--u-rampup", type=int, default=d.u_rampup)
    p.add_argument("--edl-anneal", type=int, default=d.edl_anneal)
    p.add_argument("--test-frac", type=float, default=d.test_frac)
    p.add_argument("--cal-frac", type=float, default=d.cal_frac)
    p.add_argument("--max-iterations", type=int, default=d.max_iterations,
                   help="HARD cap on optimization steps (cannot run forever)")
    p.add_argument("--max-minutes", type=float, default=d.max_minutes,
                   help="HARD wall-clock cap in minutes")
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--ckpt-every", type=int, default=d.ckpt_every)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--warmup", type=int, default=d.warmup)
    p.add_argument("--conformal-alpha", type=float, default=d.conformal_alpha)
    p.add_argument("--no-temperature", dest="temperature", action="store_false",
                   default=d.temperature,
                   help="disable temperature scaling of probabilities before conformal")
    p.add_argument("--temp-frac", type=float, default=d.temp_frac,
                   help="fraction of the cal split used to fit T (rest -> conformal)")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--max-labeled", type=int, default=d.max_labeled,
                   help="0=use all labeled train; >0=subsample labeled train to this many (low-label regime)")
    p.add_argument("--no-amp", action="store_true", default=d.no_amp)
    p.add_argument("--cudnn", choices=["auto", "on", "off"], default=d.cudnn,
                   help="cuDNN policy: auto self-tests and disables it if its 3D "
                        "kernels are unreliable on this GPU (default: %(default)s)")
    p.add_argument("--resume", action="store_true", default=d.resume)
    p.add_argument("--eval-only", action="store_true", default=d.eval_only,
                   help="skip training: load best.pt (else latest.pt) and just "
                        "re-evaluate + rewrite the results JSON")
    p.add_argument("--tag", default=d.tag, help="optional label embedded in output filenames")
    p.add_argument("--ckpt-dir", default=d.ckpt_dir,
                   help="override checkpoint dir (default logs/checkpoints); use a "
                        "unique dir per run to isolate concurrent/ablation runs")
    p.add_argument("--case-list", default=d.case_list, help="QC-clean allowlist JSON (Study 1 clean cohort)")
    p.add_argument("--n-folds", type=int, default=d.n_folds, help="patient-grouped CV folds (0=stratified single split)")
    p.add_argument("--cv-fold", type=int, default=d.cv_fold, help="which fold is the test set (0..n_folds-1)")
    p.add_argument("--cv-seed", type=int, default=d.cv_seed, help="seed for the grouped CV split")
    p.add_argument("--eval-lidc", default=d.eval_lidc, help="LIDC processed dir -> cross-site transfer AUC")
    a = p.parse_args(argv)
    return Config(**{k.replace("-", "_"): v for k, v in vars(a).items()})


def setup_logging(ts: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(LOG_DIR / f"train_{ts}.log")],
    )


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--selftest" in argv:        # hidden, isolated cuDNN probe (see configure_cudnn)
        raise SystemExit(_gpu_selftest())
    cfg = parse_args(argv)
    ts = time.strftime("%Y%m%d_%H%M%S")
    for d in (RESULTS_DIR, LOG_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    setup_logging(ts)
    seed_everything(cfg.seed)
    log.info("config: %s", json.dumps(asdict(cfg)))

    out_path = RESULTS_DIR / (f"run_{cfg.tag}_{ts}.json" if cfg.tag else f"run_{ts}.json")
    try:
        results = train(cfg, ts)
        status = 0
    except Exception as e:                          # noqa: BLE001 — always emit a report
        log.exception("training failed")
        results = {"timestamp": ts, "status": "failed", "error": repr(e),
                   "config": asdict(cfg)}
        status = 1

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    log.info("wrote results -> %s", out_path)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
