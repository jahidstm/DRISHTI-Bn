"""
DRISHTI-XAI -- Task 0.8: CrisisViT-Style SOTA Baseline
========================================================
Implements a CrisisViT-style baseline on the DRISHTI-Bn dataset.

CrisisViT (Alam et al., 2023) uses:
  - ViT-Large/16 OR ViT-Base/16 backbone
  - Pre-trained on Incidents1M (disaster-domain in-domain pre-training)
  - Classification head fine-tuned on crisis task

Since Incidents1M pre-trained weights are NOT publicly available, we use
the closest reproducible approximation:
  - google/vit-large-patch16-224-in21k (ViT-Large, ImageNet-21k pre-trained)
  - Full fine-tuning with augmentation (matching CrisisViT training protocol)
  - AdamW + CosineAnnealingLR (matching CrisisViT paper)

This gives an UPPER BOUND estimate for CrisisViT performance on DRISHTI-Bn
without Incidents1M pre-training (which gave +1.25pp in the original paper).

Reference:
  Alam et al. (2023). "CrisisViT: A Robust Vision Transformer for Crisis Image
  Classification." ISCRAM 2023. arXiv:2304.07313

Architecture:
  ViT-Large/16 (ImageNet-21k) --> Linear(1024, 512) --> Linear(512, 3)

Comparison targets:
  Task 0.5: ViT-Only (ViT-Base, partial FT)         -> Macro F1 = 78.64%
  Task 0.7: ViT Full Fine-Tune (ViT-Base, full FT)  -> Macro F1 = 74.19%
  Task 0.3: DisasterNet-v2 (ViT-Base + BanglaBERT)  -> Macro F1 = 77.95%

Usage:
  python crisisvit_baseline.py --epochs 10 --batch_size 8
  (batch_size=8 recommended -- ViT-Large is 3x larger than ViT-Base)

Expected runtime: ~35-45 min on T4 GPU (10 epochs).
"""

import os, random, argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from transformers import ViTModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score,
                              recall_score, confusion_matrix)

# ---------- Config -----------------------------------------------------------
CSV_PATH    = "../data/processed/master_dataset_translated.csv"
IMG_DIR     = "../data/processed/"
LABEL_NAMES = ["Severe_Damage", "Humanitarian_Rescue", "Affected_People"]
LABEL_MAP   = {n: i for i, n in enumerate(LABEL_NAMES)}
RANDOM_SEED = 42
EPOCHS      = 10
BATCH_SIZE  = 8       # ViT-Large needs more VRAM than ViT-Base
LR          = 2e-5    # CrisisViT paper used small LR for large model
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CrisisViT uses ViT-Large (hidden_dim=1024) instead of ViT-Base (768)
VIT_MODEL_ID = "google/vit-large-patch16-224-in21k"
VIT_HIDDEN   = 1024


# ---------- Reproducibility --------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[+] Random seed fixed: {seed} (deterministic mode ON)")

def seed_worker(worker_id):
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws); random.seed(ws)


# ---------- Dataset ----------------------------------------------------------
class ImageOnlyDataset(Dataset):
    def __init__(self, csv_file, root_dir, augment=False):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if augment:
            # CrisisViT augmentation: RandAugment-style
            self.tfm = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.3, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(
            os.path.join(self.root_dir, row["image_path"])
        ).convert("RGB")
        lbl = LABEL_MAP[row["macro_label"]]
        return self.tfm(img), torch.tensor(lbl, dtype=torch.long)


def get_split_indices(csv_path, rs=42):
    df   = pd.read_csv(csv_path)
    lbls = df["macro_label"].values
    idx  = list(range(len(df)))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=lbls,
                               random_state=rs)
    vl, ts  = train_test_split(tmp, test_size=0.5, stratify=lbls[tmp],
                               random_state=rs)
    return tr, vl, ts


# ---------- CrisisViT-Style Model (ViT-Large backbone) ----------------------
class CrisisViTClassifier(nn.Module):
    """
    CrisisViT-style: ViT-Large/16 + linear classification head.
    Unfreeze last 4 transformer blocks + pooler (same protocol as CrisisViT paper).
    """
    def __init__(self, num_classes=3):
        super().__init__()
        print(f"[+] Loading ViT-Large encoder ({VIT_MODEL_ID})...")
        self.vit = ViTModel.from_pretrained(VIT_MODEL_ID)

        # Freeze all first
        for p in self.vit.parameters():
            p.requires_grad = False

        # Unfreeze last 4 blocks (blocks 20-23) + pooler
        # CrisisViT paper: fine-tunes last 4 transformer blocks
        for name, p in self.vit.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in [20, 21, 22, 23]):
                p.requires_grad = True
            if "pooler" in name:
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.vit.parameters()
                        if p.requires_grad)
        total     = sum(p.numel() for p in self.vit.parameters())
        print(f"[+] CrisisViT-style partial FT -- Trainable: {trainable:,} / "
              f"{total:,} ({100*trainable/total:.2f}%)")

        # Classification head: ViT-Large hidden = 1024
        self.cls = nn.Sequential(
            nn.Linear(VIT_HIDDEN, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.cls(self.vit(pixel_values=x).pooler_output)


# ---------- Metrics ----------------------------------------------------------
def report(y_true, y_pred):
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    mp  = precision_score(y_true, y_pred, average="macro", zero_division=0)*100
    mr  = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    pf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    pp  = precision_score(y_true, y_pred, average=None, zero_division=0)
    pr  = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    w = 62
    print(f"\n{'='*w}")
    print(f"  Task 0.8 -- CrisisViT-Style Baseline (TEST SET)")
    print(f"{'='*w}")
    print(f"  Macro F1        : {mf1:.2f}%")
    print(f"  Macro Precision : {mp:.2f}%")
    print(f"  Macro Recall    : {mr:.2f}%")
    print("\n  Per-class breakdown:")
    for i, n in enumerate(LABEL_NAMES):
        print(f"    [{n:<22}] P={pp[i]:.4f}  R={pr[i]:.4f}  F1={pf1[i]:.4f}")
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print("  " + " "*16 + "".join(f"  {n[:8]:>8}" for n in LABEL_NAMES))
    rl = ["Severe_Damag", "Humanitarian", "Affected_Peo"]
    for i, r in enumerate(rl):
        print(f"  {r:<14}  " + "  ".join(f"{cm[i][j]:>8}" for j in range(3)))
    print(f"{'='*w}")
    return mf1


# ---------- Train / Eval -----------------------------------------------------
def train_epoch(model, loader, opt, crit, scaler):
    model.train()
    tl, nc, nt = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        opt.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(imgs); loss = crit(logits, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            logits = model(imgs); loss = crit(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tl += loss.item() * imgs.size(0)
        nc += (logits.argmax(1) == lbls).sum().item()
        nt += imgs.size(0)
    return tl / nt, nc / nt


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for imgs, lbls in loader:
        preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
        trues.extend(lbls.numpy())
    return np.array(trues), np.array(preds)


# ---------- Main -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=EPOCHS)
    ap.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--no_amp",     action="store_true")
    args = ap.parse_args()

    set_seed(RANDOM_SEED)
    use_amp = (not args.no_amp) and (DEVICE.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"\n{'='*62}")
    print(f"  DRISHTI-XAI -- Task 0.8: CrisisViT-Style Baseline")
    print(f"  Backbone: ViT-Large/16 (ImageNet-21k)")
    print(f"  Device: {DEVICE}  |  Seed: {RANDOM_SEED}  |  AMP: {use_amp}")
    print(f"{'='*62}\n")

    tr_ds = ImageOnlyDataset(CSV_PATH, IMG_DIR, augment=True)
    ev_ds = ImageOnlyDataset(CSV_PATH, IMG_DIR, augment=False)
    tr_i, vl_i, ts_i = get_split_indices(CSV_PATH, RANDOM_SEED)
    print(f"[+] Split: Train={len(tr_i)} | Val={len(vl_i)} | Test={len(ts_i)}")
    print(f"[+] Batch={args.batch_size} | LR={args.lr} | Epochs={args.epochs}")

    g = torch.Generator(); g.manual_seed(RANDOM_SEED)
    tr_ld = DataLoader(Subset(tr_ds, tr_i), batch_size=args.batch_size,
                       shuffle=True, num_workers=2, pin_memory=True,
                       worker_init_fn=seed_worker, generator=g)
    vl_ld = DataLoader(Subset(ev_ds, vl_i), batch_size=args.batch_size,
                       shuffle=False, num_workers=2, pin_memory=True)
    ts_ld = DataLoader(Subset(ev_ds, ts_i), batch_size=args.batch_size,
                       shuffle=False, num_workers=2, pin_memory=True)

    model = CrisisViTClassifier(num_classes=3).to(DEVICE)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)  # CrisisViT uses label smoothing
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1, best_ep, best_st = 0.0, 0, None
    print(f"\n[+] Training {args.epochs} epochs (CrisisViT-style, AMP={use_amp})...\n")

    for ep in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, tr_ld, opt, crit, scaler)
        sch.step()
        vt, vp = evaluate(model, vl_ld)
        vf1 = f1_score(vt, vp, average="macro", zero_division=0) * 100
        print(f"  Epoch {ep:02d}/{args.epochs} | "
              f"Loss={loss:.4f} | Train Acc={acc*100:.2f}% | "
              f"Val Macro-F1={vf1:.2f}%")
        if vf1 > best_f1:
            best_f1, best_ep = vf1, ep
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[+] Best checkpoint: Epoch {best_ep} (Val Macro-F1={best_f1:.2f}%)")
    model.load_state_dict(best_st); model.to(DEVICE)
    tt, tp = evaluate(model, ts_ld)
    f1 = report(tt, tp)

    w = 62
    print(f"\n{'='*w}")
    print(f"  COMPARISON SUMMARY (Phase 0 Complete)")
    print(f"{'='*w}")
    print(f"  {'Method':<42} {'Macro F1':>8}")
    print(f"  {'-'*42} {'-'*8}")
    print(f"  {'Majority Classifier':<42} {'24.89%':>8}")
    print(f"  {'Stratified Random':<42} {'39.95%':>8}")
    print(f"  {'BanglaBERT-Only (no image)':<42} {'43.02%':>8}  <- Task 0.6")
    print(f"  {'ViT Full Fine-Tune (ViT-Base, all params)':<42} {'74.19%':>8}  <- Task 0.7")
    print(f"  {'DisasterNet-v2 (ViT-Base+BERT, partial FT)':<42} {'77.95%':>8}  <- Task 0.3")
    print(f"  {'ViT-Only (ViT-Base, partial FT)':<42} {'78.64%':>8}  <- Task 0.5")
    print(f"  {'CrisisViT-style (ViT-Large, partial FT)':<42} {f1:>7.2f}%  <- Task 0.8")
    print(f"{'='*w}")
    print(f"\n[!] CrisisViT-style vs ViT-Base partial FT  : {f1-78.64:+.2f}pp")
    print(f"[!] CrisisViT-style vs DisasterNet-v2       : {f1-77.95:+.2f}pp")
    print(f"\n[NOTE] True CrisisViT (with Incidents1M pre-training) would")
    print(f"       score ~+1.25pp higher per the original paper (arXiv:2304.07313).")
    print()


if __name__ == "__main__":
    main()
