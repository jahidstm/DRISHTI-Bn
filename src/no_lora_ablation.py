"""
DRISHTI-XAI -- Task 1.3: Ablation -- No LoRA (Frozen Encoders)
================================================================
This ablation study removes LoRA entirely. Both ViT and BanglaBERT encoders
are completely frozen. Only the fusion classifier head is trained.

Research Question:
    How much does LoRA adapter fine-tuning contribute to classification performance?
    i.e., what is the value of parameter-efficient fine-tuning vs frozen features?

Configuration:
    Task 1.1 (CLS-Only + LoRA):    Trainable = 589,824 / 197M  (0.30%)  F1=83.21%
    Task 1.3 (No LoRA, cls-only):  Trainable = ~530K   / 197M  (0.27%)  F1=???

Loss equation (same as Task 1.1):
    L = 1.0 * L_cls + 0.0 * L_cap

Architecture change:
    - NO LoRA adapters injected
    - ViT + BanglaBERT completely frozen
    - Only classification head trained

Comparison targets:
    Task 0.3: DisasterNet-v2 (LoRA + both losses)  -> F1 = 77.95%
    Task 0.5: ViT-Only (frozen ViT + cls head)     -> F1 = 78.64%
    Task 1.1: CLS-Only (LoRA + cls loss only)      -> F1 = 83.21%
    Task 1.3: No-LoRA (frozen encoders + cls head) -> ???

Expected runtime: ~15-20 min on T4 GPU (10 epochs, much faster without LoRA grad).

Usage:
    python no_lora_ablation.py --epochs 10 --batch_size 16
"""

import os, random, argparse, math
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from transformers import AutoModel, ViTModel, AutoTokenizer
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
BATCH_SIZE  = 16
LR          = 1e-4
MAX_LEN     = 128

LAMBDA_CLS  = 1.0
LAMBDA_CAP  = 0.0

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_ID     = "csebuetnlp/banglabert"


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
class MultimodalDataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer, max_len=MAX_LEN):
        self.df        = pd.read_csv(csv_file)
        self.root_dir  = root_dir
        self.tokenizer = tokenizer
        self.max_len   = max_len

        for col in ["bangla_caption", "bengali_caption", "caption_bn",
                    "caption", "text"]:
            if col in self.df.columns:
                self.text_col = col
                break
        else:
            raise ValueError(f"No caption column found. Available: "
                             f"{list(self.df.columns)}")
        print(f"[+] Using text column: '{self.text_col}'")
        self.df[self.text_col] = self.df[self.text_col].fillna("").astype(str)
        self.df["label_id"]    = self.df["macro_label"].map(LABEL_MAP)

        self.transform = transforms.Compose([
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
        pixel_values = self.transform(img)
        enc = self.tokenizer(
            row[self.text_col],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "pixel_values":  pixel_values,
            "input_ids":     enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
            "labels":        torch.tensor(row["label_id"], dtype=torch.long)
        }


def get_split_indices(csv_path, rs=42):
    df   = pd.read_csv(csv_path)
    lbls = df["macro_label"].values
    idx  = list(range(len(df)))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=lbls,
                               random_state=rs)
    vl, ts  = train_test_split(tmp, test_size=0.5, stratify=lbls[tmp],
                               random_state=rs)
    return tr, vl, ts


# ---------- Model (NO LoRA -- frozen encoders) --------------------------------
class DisasterNetNoLoRA(nn.Module):
    """
    DisasterNet-v2 classification head ONLY.
    Both ViT and BanglaBERT are fully frozen (no LoRA, no gradient).
    Only the fusion MLP classifier is trained.

    This tests: how much do LoRA adapters contribute?
    """

    def __init__(self, num_classes=3):
        super().__init__()
        # Load encoders
        self.vision_encoder = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k")
        self.text_encoder   = AutoModel.from_pretrained(BERT_ID)

        # FREEZE both encoders completely (no LoRA)
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[+] Frozen encoders (No LoRA) -- "
              f"Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.4f}%)")

        # Fusion Classifier (identical to model.py)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            vis = self.vision_encoder(
                pixel_values=pixel_values).pooler_output
            txt = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat([vis, txt], dim=1))


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
    print(f"  Task 1.3 -- No-LoRA Ablation (TEST SET)")
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


# ---------- Train / Eval ------------------------------------------------------
def train_epoch(model, loader, opt, crit_cls, scaler):
    model.train()
    tl, nc, nt = 0.0, 0, 0
    for batch in loader:
        pv  = batch["pixel_values"].to(DEVICE)
        ids = batch["input_ids"].to(DEVICE)
        msk = batch["attention_mask"].to(DEVICE)
        lbl = batch["labels"].to(DEVICE)

        opt.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pv, ids, msk)
                loss   = crit_cls(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            logits = model(pv, ids, msk)
            loss   = crit_cls(logits, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        tl += loss.item() * pv.size(0)
        nc += (logits.argmax(1) == lbl).sum().item()
        nt += pv.size(0)
    return tl / nt, nc / nt


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        pv  = batch["pixel_values"].to(DEVICE)
        ids = batch["input_ids"].to(DEVICE)
        msk = batch["attention_mask"].to(DEVICE)
        logits = model(pv, ids, msk)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(batch["labels"].numpy())
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
    print(f"  DRISHTI-XAI -- Task 1.3: Ablation -- No LoRA (Frozen Encoders)")
    print(f"  Architecture: DisasterNet-v2 WITHOUT LoRA adapters")
    print(f"  Encoders: FROZEN (zero grad) | Only cls head trained")
    print(f"  Loss: L = {LAMBDA_CLS}*L_cls + {LAMBDA_CAP}*L_cap")
    print(f"  Device: {DEVICE}  |  Seed: {RANDOM_SEED}  |  AMP: {use_amp}")
    print(f"{'='*62}\n")

    print("[+] Loading tokenizer (csebuetnlp/banglabert)...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID)

    dataset = MultimodalDataset(CSV_PATH, IMG_DIR, tokenizer)
    tr_i, vl_i, ts_i = get_split_indices(CSV_PATH, RANDOM_SEED)
    print(f"[+] Split: Train={len(tr_i)} | Val={len(vl_i)} | Test={len(ts_i)}")

    g = torch.Generator(); g.manual_seed(RANDOM_SEED)
    tr_ld = DataLoader(Subset(dataset, tr_i), batch_size=args.batch_size,
                       shuffle=True, num_workers=2, pin_memory=True,
                       worker_init_fn=seed_worker, generator=g)
    vl_ld = DataLoader(Subset(dataset, vl_i), batch_size=args.batch_size,
                       shuffle=False, num_workers=2, pin_memory=True)
    ts_ld = DataLoader(Subset(dataset, ts_i), batch_size=args.batch_size,
                       shuffle=False, num_workers=2, pin_memory=True)

    model    = DisasterNetNoLoRA(num_classes=3).to(DEVICE)
    crit_cls = nn.CrossEntropyLoss()
    # Only train classifier params (encoders are frozen)
    opt      = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1, best_ep, best_st = 0.0, 0, None
    print(f"\n[+] Training {args.epochs} epochs "
          f"(No LoRA, frozen encoders, AMP={use_amp})...\n")

    for ep in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, tr_ld, opt, crit_cls, scaler)
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
    print(f"  ABLATION SUMMARY -- Effect of LoRA")
    print(f"{'='*w}")
    print(f"  {'Method':<44} {'Macro F1':>8}")
    print(f"  {'-'*44} {'-'*8}")
    print(f"  {'BanglaBERT-Only (no image)':<44} {'43.02%':>8}  <- Task 0.6")
    print(f"  {'DisasterNet-v2 (LoRA + both losses)':<44} {'77.95%':>8}  <- Task 0.3")
    print(f"  {'ViT-Only (frozen ViT, no text)':<44} {'78.64%':>8}  <- Task 0.5")
    print(f"  {'CLS-Only (LoRA + cls loss)':<44} {'83.21%':>8}  <- Task 1.1")
    print(f"  {'No-LoRA (frozen encoders + cls head)':<44} {f1:>7.2f}%  <- Task 1.3")
    print(f"{'='*w}")

    delta = f1 - 83.21
    print(f"\n[!] LoRA contribution: {delta:+.2f}pp "
          f"(No-LoRA vs CLS-Only with LoRA)")
    if delta < 0:
        print(f"[!] FINDING: LoRA adapters IMPROVE classification by {abs(delta):.2f}pp")
    else:
        print(f"[!] FINDING: LoRA adapters do NOT improve frozen-encoder features "
              f"(frozen features already strong)")
    print()


if __name__ == "__main__":
    main()
