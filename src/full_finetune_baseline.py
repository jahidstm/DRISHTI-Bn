"""
DRISHTI-XAI -- Task 0.7: Full Fine-Tune Baseline (ViT, No Frozen Layers, No LoRA)
==================================================================================
Trains a FULLY fine-tuned ViT classifier on DRISHTI-Bn.
ALL ViT parameters are trainable (no frozen blocks, no LoRA).

Architecture:
    ViT-base-patch16-224 (ALL params trainable) -> Linear(768,512) -> Linear(512,3)

Comparison targets:
    Task 0.3: DisasterNet-v2 (ViT+BanglaBERT, partial FT) -> Macro F1 = 77.95%
    Task 0.5: ViT-Only (last 2 blocks unfrozen)            -> Macro F1 = 78.64%
    Task 0.6: BanglaBERT-Only (last 4 layers unfrozen)     -> Macro F1 = 43.02%

Usage:
    python full_finetune_baseline.py --epochs 10 --batch_size 16

Note: Full FT uses ~86M params vs ~7M in partial FT.
      batch_size=16 recommended for T4 GPU.
      Expected runtime: ~20-25 min (10 epochs, T4).
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
BATCH_SIZE  = 16    # lower default -- full FT is more VRAM-heavy
LR          = 5e-5  # lower LR to avoid catastrophic forgetting
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            self.tfm = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root_dir, row["image_path"])).convert("RGB")
        lbl = LABEL_MAP[row["macro_label"]]
        return self.tfm(img), torch.tensor(lbl, dtype=torch.long)


def get_split_indices(csv_path, rs=42):
    df   = pd.read_csv(csv_path)
    lbls = df["macro_label"].values
    idx  = list(range(len(df)))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=lbls, random_state=rs)
    vl, ts  = train_test_split(tmp, test_size=0.5, stratify=lbls[tmp], random_state=rs)
    return tr, vl, ts


# ---------- Model (ALL params unfrozen) -------------------------------------
class ViTFullFT(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        print("[+] Loading ViT encoder (google/vit-base-patch16-224-in21k)...")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # Unfreeze EVERYTHING
        for p in self.vit.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.vit.parameters())
        print(f"[+] Full fine-tune -- Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.2f}%)")
        self.cls = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.cls(self.vit(pixel_values=x).pooler_output)


# ---------- Metrics ----------------------------------------------------------
def report(y_true, y_pred):
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    mp  = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    mr  = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    pf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    pp  = precision_score(y_true, y_pred, average=None, zero_division=0)
    pr  = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    w = 62
    print(f"\n{'='*w}")
    print(f"  Task 0.7 -- Full Fine-Tune Baseline (TEST SET)")
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


# ---------- Train / Eval loops -----------------------------------------------
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
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\n{'='*62}")
    print(f"  DRISHTI-XAI -- Task 0.7: Full Fine-Tune Baseline (ViT)")
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

    model = ViTFullFT(num_classes=3).to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch   = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, steps_per_epoch=len(tr_ld),
        epochs=args.epochs, pct_start=0.1, anneal_strategy="cos"
    )

    best_f1, best_ep, best_st = 0.0, 0, None
    print(f"\n[+] Training {args.epochs} epochs (full FT, AMP={use_amp})...\n")

    for ep in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, tr_ld, opt, crit, scaler)
        sch.step()
        vt, vp = evaluate(model, vl_ld)
        vf1 = f1_score(vt, vp, average="macro", zero_division=0) * 100
        print(f"  Epoch {ep:02d}/{args.epochs} | "
              f"Loss={loss:.4f} | Train Acc={acc*100:.2f}% | Val Macro-F1={vf1:.2f}%")
        if vf1 > best_f1:
            best_f1, best_ep = vf1, ep
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[+] Best checkpoint: Epoch {best_ep} (Val Macro-F1={best_f1:.2f}%)")
    model.load_state_dict(best_st); model.to(DEVICE)
    tt, tp = evaluate(model, ts_ld)
    f1 = report(tt, tp)

    w = 62
    print(f"\n{'='*w}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*w}")
    print(f"  {'Method':<38} {'Macro F1':>8}")
    print(f"  {'-'*38} {'-'*8}")
    print(f"  {'Majority Classifier':<38} {'24.89%':>8}")
    print(f"  {'Stratified Random':<38} {'39.95%':>8}")
    print(f"  {'BanglaBERT-Only (no image)':<38} {'43.02%':>8}  <- Task 0.6")
    print(f"  {'DisasterNet-v2 (ViT+BERT, partial FT)':<38} {'77.95%':>8}  <- Task 0.3")
    print(f"  {'ViT-Only (partial FT, last 2 blocks)':<38} {'78.64%':>8}  <- Task 0.5")
    print(f"  {'ViT Full Fine-Tune (this script)':<38} {f1:>7.2f}%  <- Task 0.7")
    print(f"{'='*w}")
    print(f"\n[!] Full FT vs Partial FT  : {f1-78.64:+.2f}pp")
    print(f"[!] Full FT vs DisasterNet : {f1-77.95:+.2f}pp")
    print()


if __name__ == "__main__":
    main()
