"""
DRISHTI-XAI — Task 0.5: ViT-Only Baseline
==========================================
Trains and evaluates a Vision-Transformer-ONLY classifier on the DRISHTI-Bn
dataset. NO text encoder (BanglaBERT) is used. This isolates the contribution
of visual features alone.

Architecture:
    ViT-base-patch16-224 (LoRA r=8)  →  Linear(768, 512)  →  Linear(512, 3)

Comparison target (Task 0.3):
    DisasterNet-v2 (ViT + BanglaBERT + LoRA)  →  Macro F1 = 77.95%

Usage (run from /content/DRISHTI-Bn/src on Colab):
    python vit_only_baseline.py

Expected runtime: ~10-12 min on T4 GPU (10 epochs, 376 test samples)
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from transformers import ViTModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = '../data/processed/master_dataset_translated.csv'
IMG_DIR     = '../data/processed/'
LABEL_NAMES = ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']
LABEL_MAP   = {name: i for i, name in enumerate(LABEL_NAMES)}
RANDOM_SEED = 42
EPOCHS      = 10
BATCH_SIZE  = 32
LR          = 2e-4
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Fix all random seeds for full reproducibility (standard academic practice)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[+] Random seed fixed: {seed} (deterministic mode ON)")

def seed_worker(worker_id):
    """Worker seed initialiser for DataLoader reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ── Dataset (Image-only, no text) ─────────────────────────────────────────────
class ImageOnlyDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = LABEL_MAP[row['macro_label']]
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def get_split_indices(csv_path, random_state=42):
    """Identical 80/10/10 stratified split as data_loader.py."""
    df = pd.read_csv(csv_path)
    labels = df['macro_label'].values
    all_idx = list(range(len(df)))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.2, stratify=labels, random_state=random_state
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=random_state
    )
    return train_idx, val_idx, test_idx


# ── Model (ViT + LoRA + Classification Head) ──────────────────────────────────
class ViTOnlyClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        print("[+] Loading ViT encoder (google/vit-base-patch16-224-in21k)...")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # Freeze all ViT layers first
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last 2 transformer blocks + pooler (by name matching)
        for name, param in self.vit.named_parameters():
            if ("encoder.layer.10" in name or
                "encoder.layer.11" in name or
                "pooler" in name):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.vit.parameters())
        print(f"[+] Partial fine-tune — Trainable: {trainable:,} / {total:,} params "
              f"({100*trainable/total:.2f}%)")

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled  = outputs.pooler_output          # [B, 768]
        return self.classifier(pooled)


# ── Metrics helper ────────────────────────────────────────────────────────────
def print_cls_report(y_true, y_pred, title="ViT-Only Classifier"):
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    macro_p  = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    macro_r  = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    per_f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_p    = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_r    = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm       = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    w = 62
    print(f"\n{'='*w}")
    print(f"  [CLS] {title}")
    print(f"{'='*w}")
    print(f"  Macro F1        : {macro_f1:.2f}%")
    print(f"  Macro Precision : {macro_p:.2f}%")
    print(f"  Macro Recall    : {macro_r:.2f}%")
    print()
    print("  Per-class breakdown:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"    [{name:<22}] P={per_p[i]:.4f}  R={per_r[i]:.4f}  F1={per_f1[i]:.4f}")
    print()
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':>16}" + "".join(f"  {n[:8]:>8}" for n in LABEL_NAMES))
    row_labels = ["Severe_Damag", "Humanitarian", "Affected_Peo"]
    for i, rl in enumerate(row_labels):
        row_str = "  ".join(f"{cm[i][j]:>8}" for j in range(3))
        print(f"  {rl:<14}  {row_str}")
    print(f"{'='*w}")
    return macro_f1


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(RANDOM_SEED)   # ← deterministic reproducibility

    print(f"\n{'='*62}")
    print(f"  DRISHTI-XAI -- Task 0.5: ViT-Only Baseline (No Text)")
    print(f"  Device: {DEVICE}  |  Seed: {RANDOM_SEED}")
    print(f"{'='*62}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    dataset = ImageOnlyDataset(CSV_PATH, IMG_DIR)
    train_idx, val_idx, test_idx = get_split_indices(CSV_PATH, RANDOM_SEED)

    print(f"[+] Split: Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model     = ViTOnlyClassifier(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training ──────────────────────────────────────────────────────────
    best_val_f1, best_epoch = 0.0, 0
    best_state = None

    print(f"\n[+] Training for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_true, val_pred    = evaluate(model, val_loader)
        val_f1 = f1_score(val_true, val_pred, average='macro', zero_division=0) * 100
        scheduler.step()

        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss={train_loss:.4f} | Train Acc={train_acc*100:.2f}% | "
              f"Val Macro-F1={val_f1:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[+] Best checkpoint: Epoch {best_epoch} (Val Macro-F1={best_val_f1:.2f}%)")

    # ── Test evaluation on best checkpoint ────────────────────────────────
    model.load_state_dict(best_state)
    model.to(DEVICE)

    test_true, test_pred = evaluate(model, test_loader)
    f1 = print_cls_report(test_true, test_pred,
                          title="Task 0.5 — ViT-Only Baseline (TEST SET)")

    # ── Comparison summary ────────────────────────────────────────────────
    w = 62
    print(f"\n{'='*w}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*w}")
    print(f"  {'Method':<36} {'Macro F1':>8}")
    print(f"  {'-'*36} {'-'*8}")
    print(f"  {'Majority Classifier':<36} {'24.89%':>8}")
    print(f"  {'Stratified Random':<36} {'39.95%':>8}")
    print(f"  {'ViT-Only (this script)':<36} {f1:>7.2f}%  <- Task 0.5")
    print(f"  {'DisasterNet-v2 (ViT + BanglaBERT)':<36} {'77.95%':>8}  <- Task 0.3")
    print(f"{'='*w}")
    print(f"\n[!] Text encoder contribution: {77.95 - f1:+.2f}pp improvement over ViT-Only")
    print()


if __name__ == '__main__':
    main()
