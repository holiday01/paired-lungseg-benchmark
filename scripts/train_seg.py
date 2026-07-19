#!/usr/bin/env python3
"""Semi-supervised 3D nodule/tumor segmentation — supervised vs Mean Teacher.

Labeled  = cancer tumor masks (data/processed_seg/*.npz, split by case).
Unlabeled= the img_nodule pool (data/processed/img_nodule/*.npz, image only).

--ssl none        : supervised-only baseline (labeled patches, DiceCE).
--ssl meanteacher : + EMA teacher consistency (MSE) on unlabeled patches, ramp-up.

Metric = foreground (tumor) Dice via sliding-window inference on held-out cases.
Writes results/seg_<tag>_<ts>.json. Bounded, reproducible, cudnn-safe (native 3D).
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys, time, copy, numpy as np, torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import train as T   # reuse configure_cudnn, seed_everything
from monai.networks.nets import (BasicUNet, SegResNet, AttentionUnet, UNet,
                                  DynUNet, UNETR, MedNeXt, VNet)
from monai.losses import DiceCELoss, TverskyLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

SEG_DIR = ROOT / "data" / "processed_seg"
POOL_DIR = ROOT / "data" / "processed" / "img_nodule"
RESULTS = ROOT / "results"
PATCH = (96, 96, 96)
HU_MIN, HU_MAX = -1000.0, 400.0


def norm(img):
    return ((np.clip(img, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


_CLEAN_MIN = 0   # set from --clean-mask-min; remove tumor-mask components smaller than this many voxels


def _clean_small(m, min_vox):
    """Drop connected components smaller than min_vox (removes scattered single-voxel label noise)."""
    if min_vox <= 0 or m.sum() == 0:
        return m
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (sizes >= min_vox)[lab].astype(np.uint8)


def load_seg(p):
    d = np.load(p, allow_pickle=True)
    return norm(d["image"]), _clean_small(d["mask"].astype(np.uint8), _CLEAN_MIN), str(d["case"])


def rand_patch(img, mask=None, fg_prob=0.8, rng=None):
    rng = rng or np.random
    sh = img.shape
    pad = [max(0, P - s) for P, s in zip(PATCH, sh)]
    if any(pad):
        img = np.pad(img, [(0, p) for p in pad], constant_values=0.0)
        if mask is not None:
            mask = np.pad(mask, [(0, p) for p in pad], constant_values=0)
        sh = img.shape
    if mask is not None and mask.any() and rng.random() < fg_prob:
        fg = np.argwhere(mask > 0); c = fg[rng.randint(len(fg))]
        start = [int(np.clip(c[i] - PATCH[i] // 2, 0, sh[i] - PATCH[i])) for i in range(3)]
    else:
        start = [rng.randint(0, sh[i] - PATCH[i] + 1) for i in range(3)]
    sl = tuple(slice(start[i], start[i] + PATCH[i]) for i in range(3))
    return (img[sl], None if mask is None else mask[sl])


def np_augment(img, msk, rng):
    """Lightweight on-patch augmentation (verified-missing fix): flips, 90deg rot, gamma, noise, intensity.
    img float patch, msk uint8 patch. Geometric aug applied to BOTH; intensity only to img."""
    # random flips on all 3 axes
    for ax in range(3):
        if rng.random() < 0.5:
            img = np.flip(img, ax); msk = np.flip(msk, ax)
    # random 90-deg rotation in a random plane
    if rng.random() < 0.5:
        ax = tuple(rng.choice(3, 2, replace=False)); k = rng.randint(1, 4)
        img = np.rot90(img, k, ax); msk = np.rot90(msk, k, ax)
    img = np.ascontiguousarray(img); msk = np.ascontiguousarray(msk)
    # intensity: gamma, scale/shift, gaussian noise (img is normalized ~[0,1])
    if rng.random() < 0.3:
        g = rng.uniform(0.7, 1.5); img = np.clip(img, 0, None) ** g
    if rng.random() < 0.3:
        img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
    if rng.random() < 0.2:
        img = img + rng.normal(0, 0.03, img.shape).astype(img.dtype)
    return img.astype(np.float32), msk.astype(np.uint8)


def build_tumor_lib(train_lab):
    """Tight tumor crops (image, mask) from labeled volumes, for copy-paste augmentation."""
    lib = []
    for img, msk, _ in train_lab:
        if msk.sum() == 0:
            continue
        nz = np.argwhere(msk > 0); lo = nz.min(0); hi = nz.max(0) + 1
        sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
        lib.append((img[sl].copy(), msk[sl].copy()))
    return lib


def paste_lesion(img, mask, lib, rng, spacing, mm_range=(4, 12)):
    """CarveMix/BCP-style: carve a labeled tumor, downscale to a small diameter, paste it
    at a random in-lung location. Manufactures small-target supervision to bridge the
    large-mass(labeled) -> tiny-nodule(deployment) gap. Mutates and returns (img, mask)."""
    from scipy.ndimage import zoom
    ti, tm = lib[rng.randint(len(lib))]
    cur_mm = ((tm.sum() * 6 / np.pi) ** (1 / 3)) * spacing          # equiv diameter
    z = max(0.05, float(rng.uniform(*mm_range)) / max(cur_mm, 1e-3))
    z = min(z, *(0.9 * img.shape[i] / max(tm.shape[i], 1) for i in range(3)))
    ti2 = zoom(ti, z, order=1); tm2 = zoom(tm.astype(np.float32), z, order=0) > 0.5
    ts = tm2.shape
    if tm2.sum() == 0 or any(ts[i] >= img.shape[i] for i in range(3)):
        return img, mask
    for _ in range(8):                                              # try to land in lung tissue
        pos = [rng.randint(0, img.shape[i] - ts[i] + 1) for i in range(3)]
        sl = tuple(slice(pos[i], pos[i] + ts[i]) for i in range(3))
        if img[sl][tm2].mean() > 0.07:                             # ~ > -900 HU (in-lung)
            break
    reg = img[sl].copy(); reg[tm2] = ti2[tm2]; img[sl] = reg
    m = mask[sl].copy(); m[tm2] = 1; mask[sl] = m
    return img, mask


def _enable_dropout(model):
    """Enable only Dropout layers (MC-dropout for UA-MT uncertainty); norm stays in eval."""
    for mod in model.modules():
        if mod.__class__.__name__.startswith("Dropout"):
            mod.train()


def ema_update(teacher, student, decay):
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.mul_(decay).add_(sp.detach(), alpha=1 - decay)
        for tb, sb in zip(teacher.buffers(), student.buffers()):
            tb.copy_(sb)


def evaluate(model, cases, device, postproc=False):
    """Foreground Dice via sliding-window over each held-out case volume."""
    metric = DiceMetric(include_background=False, reduction="mean")
    model.eval()
    dices, sizes = [], []
    with torch.no_grad():
        for img, msk, _ in cases:
            if msk.sum() == 0:           # honest: skip empty-GT cases (no Dice=1.0 freebie)
                continue
            x = torch.from_numpy(img)[None, None].to(device)
            logits = sliding_window_inference(x, PATCH, 2, model, overlap=0.5, mode="gaussian")
            pred = logits.argmax(1, keepdim=True).float()
            if postproc:     # keep largest connected component (FP cleanup)
                from scipy.ndimage import label as cc_label
                pa = pred[0, 0].cpu().numpy().astype(np.uint8)
                if pa.sum() > 0:
                    lab, n = cc_label(pa)
                    if n > 1:
                        big = 1 + int(np.argmax([(lab == k).sum() for k in range(1, n + 1)]))
                        pa = (lab == big).astype(np.uint8)
                    pred = torch.from_numpy(pa)[None, None].float().to(device)
            y = torch.from_numpy(msk)[None, None].float().to(device)
            metric(y_pred=pred, y=y)
            inter = float((pred.bool() & y.bool()).sum())
            denom = float(pred.sum() + y.sum())
            dices.append(2 * inter / denom if denom > 0 else 0.0)
            sizes.append(float(msk.sum()))
    try:
        agg = float(metric.aggregate().item())
    except Exception:
        agg = float("nan")
    metric.reset()
    # stash per-case detail on the function for size-stratified + detection reporting
    evaluate.last = {"dices": dices, "sizes_vox": sizes}
    return float(np.mean(dices)), agg          # primary = manual per-volume Dice


def main():
    global SEG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl", choices=["none", "meanteacher", "uamt", "pseudo"], default="none")
    ap.add_argument("--model", choices=["basicunet", "segresnet", "attentionunet", "unet", "swinunetr",
                                        "dynunet", "unetr", "mednext", "vnet", "unetr_lite"],
                    default="basicunet")
    ap.add_argument("--max-iterations", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--labeled-bs", type=int, default=2)
    ap.add_argument("--unlabeled-bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cons-weight", type=float, default=1.0)
    ap.add_argument("--cons-rampup", type=int, default=400)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--uamt-t", type=int, default=4, help="UA-MT: MC-dropout forward passes for uncertainty")
    ap.add_argument("--dropout", type=float, default=0.0, help="model dropout (>0 enables MC-dropout for UA-MT)")
    ap.add_argument("--pseudo-thresh", type=float, default=0.9, help="confidence threshold for pseudo-labeling SSL")
    ap.add_argument("--patience", type=int, default=0, help="early stop after this many consecutive evals with no val-Dice improvement (0=off)")
    ap.add_argument("--loss", choices=["dicece", "focaltversky"], default="dicece")
    ap.add_argument("--ema-eval", action="store_true", help="use EMA weights for eval even when ssl=none")
    ap.add_argument("--copypaste", action="store_true", help="CarveMix/BCP small-lesion copy-paste (OOD bridge)")
    ap.add_argument("--augment", action="store_true", help="supervised data augmentation (flip/rot90/gamma/noise) — verified-missing fix")
    ap.add_argument("--postproc", action="store_true", help="keep largest connected component at eval (FP cleanup)")
    ap.add_argument("--spacing", type=float, default=1.0, help="data isotropic mm (for lesion mm scaling)")
    ap.add_argument("--n-unlabeled", type=int, default=400, help="pool volumes to use")
    ap.add_argument("--max-labeled-cases", type=int, default=0,
                    help="0=use all labeled train cases; >0=cap to this many (low-label regime)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--seg-dir", default=str(SEG_DIR), help="labeled seg dir (e.g. processed_seg_1mm)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--case-list", default="", help="JSON QC allowlist of good seg-case filenames; '' = use all")
    ap.add_argument("--clean-mask-min", type=int, default=0, help="remove tumor-mask components < N voxels (0=off)")
    cfg = ap.parse_args()
    SEG_DIR = Path(cfg.seg_dir)
    global _CLEAN_MIN; _CLEAN_MIN = cfg.clean_mask_min

    T.seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T.configure_cudnn("auto", device)
    rng = np.random.RandomState(cfg.seed)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # --- data: split labeled by CASE ---
    files = sorted(glob.glob(str(SEG_DIR / "*.npz")))
    if cfg.case_list:
        cl = json.load(open(cfg.case_list))
        good = set(cl["sets"]["seg_1mm"]["good"]) if "sets" in cl else set(cl.get("good", cl))
        before = len(files)
        files = [f for f in files if os.path.basename(f) in good]
        print(f"case-list filter ({cfg.case_list}): kept {len(files)}/{before} seg files")
    cases = {}
    for f in files:
        c = str(np.load(f, allow_pickle=True)["case"]); cases.setdefault(c, []).append(f)
    case_ids = sorted(cases); rng.shuffle(case_ids)
    n = len(case_ids); n_test = max(1, n // 5); n_val = max(1, n // 5)
    test_c, val_c, train_c = case_ids[:n_test], case_ids[n_test:n_test+n_val], case_ids[n_test+n_val:]
    # Low-label regime: cap labeled TRAIN cases (val/test held fixed for comparability across budgets).
    if cfg.max_labeled_cases and cfg.max_labeled_cases < len(train_c):
        train_c = train_c[:cfg.max_labeled_cases]
        print(f"low-label regime: capped labeled train to {len(train_c)} cases")
    def load_cases(cs): return [load_seg(f) for c in cs for f in cases[c]]
    train_lab = load_cases(train_c); val_set = load_cases(val_c); test_set = load_cases(test_c)
    print(f"cases: train={len(train_c)} val={len(val_c)} test={len(test_c)} | "
          f"labeled vols train={len(train_lab)} val={len(val_set)} test={len(test_set)}")

    pool_files = sorted(glob.glob(str(POOL_DIR / "*.npz")))[:cfg.n_unlabeled]
    pool = [norm(np.load(f)["image"]) for f in pool_files] if (cfg.ssl != "none" or cfg.copypaste) else []
    print(f"unlabeled pool volumes: {len(pool)}  ssl={cfg.ssl}")

    # --- model(s) ---
    def make():
        m = cfg.model
        if m == "basicunet":
            net = BasicUNet(spatial_dims=3, in_channels=1, out_channels=2,
                            features=(32, 32, 64, 128, 256, 32))
        elif m == "segresnet":
            net = SegResNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16,
                            blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
        elif m == "attentionunet":
            net = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=2,
                                channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
        elif m == "unet":
            net = UNet(spatial_dims=3, in_channels=1, out_channels=2,
                       channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
        elif m == "swinunetr":
            from monai.networks.nets import SwinUNETR
            try:
                net = SwinUNETR(in_channels=1, out_channels=2, feature_size=24, spatial_dims=3)
            except TypeError:
                net = SwinUNETR(img_size=PATCH, in_channels=1, out_channels=2, feature_size=24)
        # New models (2023-2024)
        elif m == "dynunet":
            net = DynUNet(spatial_dims=3, in_channels=1, out_channels=2,
                         kernel_size=[3, 3, 3, 3, 3], strides=[1, 2, 2, 2, 2],
                         upsample_kernel_size=[2, 2, 2, 2], filters=[16, 32, 64, 128, 256],
                         dropout=cfg.dropout)
        elif m == "unetr":
            net = UNETR(in_channels=1, out_channels=2, img_size=PATCH,
                       feature_size=16, hidden_size=768, mlp_dim=3072,
                       num_heads=12, proj_type='conv', norm_name='instance', spatial_dims=3)
        elif m == "mednext":
            net = MedNeXt(in_channels=1, out_channels=2, spatial_dims=3)
        elif m == "vnet":
            net = VNet(spatial_dims=3, in_channels=1, out_channels=2, dropout_prob=0.3)
        elif m == "unetr_lite":
            net = UNETR(in_channels=1, out_channels=2, img_size=PATCH,
                       feature_size=8, hidden_size=256, mlp_dim=1024,
                       num_heads=4, proj_type='conv', norm_name='instance', spatial_dims=3)
        else:
            raise ValueError(f"unknown model {m}")
        return net.to(device)
    student = make()
    # Always keep an EMA copy: it is the Mean-Teacher pseudo-label source AND the
    # inference model (EMA-eval gives a free Dice bump even for supervised — Expert rec).
    ema_model = copy.deepcopy(student)
    for p in ema_model.parameters(): p.requires_grad_(False)
    ema_model.eval()
    teacher = ema_model if cfg.ssl in ("meanteacher", "uamt", "pseudo") else None   # pseudo-label source
    eval_model = ema_model if (cfg.ssl in ("meanteacher", "uamt", "pseudo") or cfg.ema_eval) else student
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_iterations)
    # focal-Tversky (alpha<beta) penalises false negatives -> recall on small lesions.
    if cfg.loss == "focaltversky":
        dce = DiceCELoss(to_onehot_y=True, softmax=True)
        tv = TverskyLoss(to_onehot_y=True, softmax=True, alpha=0.3, beta=0.7)
        seg_loss = lambda p, y: dce(p, y) + tv(p, y)
    else:
        seg_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    tumor_lib = build_tumor_lib(train_lab) if cfg.copypaste else []

    def lab_batch():
        xs, ys = [], []
        for _ in range(cfg.labeled_bs):
            img, msk, _ = train_lab[rng.randint(len(train_lab))]
            # copy-paste onto a POOL canvas (BCP-style OOD bridge) with prob 0.5
            if cfg.copypaste and pool and rng.random() < 0.5:
                pi = pool[rng.randint(len(pool))]; pi, _ = rand_patch(pi, None, rng=rng)
                pi = pi.copy(); pm = np.zeros(pi.shape, np.uint8)
            else:
                pi, pm = rand_patch(img, msk, rng=rng); pi, pm = pi.copy(), pm.copy()
            if cfg.copypaste and tumor_lib:
                for _ in range(rng.randint(1, 3)):
                    pi, pm = paste_lesion(pi, pm, tumor_lib, rng, cfg.spacing)
            if cfg.augment:
                pi, pm = np_augment(pi, pm, rng)
            xs.append(pi); ys.append(pm)
        return (torch.from_numpy(np.stack(xs))[:, None].to(device),
                torch.from_numpy(np.stack(ys).astype(np.int64))[:, None].to(device))

    def unlab_batch():
        xs = []
        for _ in range(cfg.unlabeled_bs):
            img = pool[rng.randint(len(pool))]; pi, _ = rand_patch(img, None, rng=rng); xs.append(pi)
        return torch.from_numpy(np.stack(xs))[:, None].to(device)

    best_dice, hist, t0 = -1.0, [], time.time()
    evals_since_best = 0
    best_path = SEG_DIR / f"best_{cfg.tag or cfg.ssl}.pt"
    for it in range(cfg.max_iterations):
        student.train()
        xb, yb = lab_batch()
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = student(xb)
            loss_sup = seg_loss(logits, yb)
            loss_cons = torch.zeros((), device=device)
            if teacher is not None:
                xu = unlab_batch()
                noise = torch.randn_like(xu) * 0.05
                if cfg.ssl == "pseudo":
                    with torch.no_grad():
                        pt = torch.softmax(teacher(xu), 1)
                        conf, plab = pt.max(1)                      # hard pseudo-label + confidence
                        pmask = (conf > cfg.pseudo_thresh).float()  # keep only confident voxels
                    ce = F.cross_entropy(student(xu + noise), plab, reduction="none")
                    loss_cons = (pmask * ce).sum() / (pmask.sum() + 1e-6)
                else:
                    mask = None
                    with torch.no_grad():
                        if cfg.ssl == "uamt":
                            _enable_dropout(teacher)
                            pt = torch.stack([torch.softmax(teacher(xu), 1)
                                              for _ in range(cfg.uamt_t)]).mean(0)
                            teacher.eval()
                            ent = -(pt * torch.log(pt + 1e-6)).sum(1, keepdim=True)
                            thr = (0.5 + 0.5 * min(1.0, (it + 1) / cfg.max_iterations)) * math.log(2.0)
                            mask = (ent < thr).float()
                        else:
                            pt = torch.softmax(teacher(xu), 1)
                    ps = torch.softmax(student(xu + noise), 1)
                    if mask is not None:
                        se = ((ps - pt) ** 2).mean(1, keepdim=True)
                        loss_cons = (mask * se).sum() / (mask.sum() + 1e-6)
                    else:
                        loss_cons = F.mse_loss(ps, pt)
            w = cfg.cons_weight * min(1.0, (it + 1) / max(1, cfg.cons_rampup))
            loss = loss_sup + w * loss_cons
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        ema_update(ema_model, student, cfg.ema_decay)

        if (it + 1) % 50 == 0:
            print(f"it {it+1}/{cfg.max_iterations} sup={float(loss_sup):.3f} "
                  f"cons={float(loss_cons):.4f} w={w:.2f} ({(time.time()-t0)/(it+1):.2f}s/it)", flush=True)
        if (it + 1) % cfg.eval_every == 0:
            d_monai, d_manual = evaluate(eval_model, val_set, device, cfg.postproc)
            hist.append({"iter": it + 1, "val_dice": d_monai})
            print(f"[eval] it {it+1} val_dice={d_monai:.4f}", flush=True)
            if d_monai > best_dice:
                best_dice = d_monai; torch.save(eval_model.state_dict(), best_path)
                evals_since_best = 0
            else:
                evals_since_best += 1
                if cfg.patience and evals_since_best >= cfg.patience:
                    print(f"[early-stop] no val improvement for {cfg.patience} evals "
                          f"(best={best_dice:.4f}); stopping at it {it+1}", flush=True)
                    break

    # --- final test with best checkpoint ---
    if best_path.exists():
        eval_model.load_state_dict(torch.load(best_path, map_location=device))
    test_dice, test_dice_manual = evaluate(eval_model, test_set, device, cfg.postproc)
    # size-stratified Dice + lesion detection sensitivity (honest annotation-accuracy metrics)
    det = evaluate.last
    dd = np.array(det["dices"]); sz_mm3 = np.array(det["sizes_vox"]) * (cfg.spacing ** 3) / 1000.0
    bands = {"<1cm3": sz_mm3 < 1, "1-10cm3": (sz_mm3 >= 1) & (sz_mm3 < 10),
             "10-50cm3": (sz_mm3 >= 10) & (sz_mm3 < 50), ">=50cm3": sz_mm3 >= 50}
    dice_by_size = {k: (float(dd[m].mean()) if m.any() else None, int(m.sum())) for k, m in bands.items()}
    detection = float((dd > 0.1).mean())   # lesion found (any meaningful overlap) — annotation-assist value
    out = {"timestamp": ts, "ssl": cfg.ssl, "tag": cfg.tag, "seed": cfg.seed,
           "n_train_cases": len(train_c), "n_val_cases": len(val_c), "n_test_cases": len(test_c),
           "n_unlabeled": len(pool), "max_iterations": cfg.max_iterations,
           "best_val_dice": best_dice, "test_dice": test_dice,
           "dice_by_size": dice_by_size, "lesion_detection_sens": detection,
           "history": hist, "config": vars(cfg)}
    print(f"  size-stratified Dice: " + " ".join(f"{k}={v[0]:.3f}(n{v[1]})" if v[0] is not None else f"{k}=NA(n{v[1]})" for k,v in dice_by_size.items()))
    print(f"  lesion detection sensitivity (Dice>0.1): {detection:.3f}")
    RESULTS.mkdir(exist_ok=True)
    jp = RESULTS / f"seg_{cfg.tag or cfg.ssl}_{ts}.json"
    json.dump(out, open(jp, "w"), indent=2)
    print(f"\nTEST tumor Dice = {test_dice:.4f}  (ssl={cfg.ssl})\nwrote {jp}")


if __name__ == "__main__":
    main()
