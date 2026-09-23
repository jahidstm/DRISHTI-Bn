"""
DRISHTI-XAI — Task 0.6: BanglaBERT-Only Baseline
==================================================
Trains and evaluates a BanglaBERT-ONLY classifier on the DRISHTI-Bn
dataset. NO image encoder (ViT) is used. Uses only Bangla captions.
This isolates the contribution of textual features alone.

Architecture:
    BanglaBERT (csebuetnlp/banglabert) → [CLS] pooling → Linear(768, 512) → Linear(512, 3)

Comparison targets:
    Task 0.3  DisasterNet-v2 (ViT + BanglaBERT)   → Macro F1 = 77.95%
    Task 0.5  ViT-Only                             → Macro F1 = 78.64%

Usage (run from /content/DRISHTI-Bn/src on Colab):
    python banglabert_only_baseline.py
    python banglabert_only_baseline.py --epochs 10 --batch_size 16

Expected runtime: ~8-12 min on T4 GPU (10 epochs, 376 test samples)
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# -- Config -------------------------------------------------------------------
CSV_PATH      = '../data/processed/master_dataset_translated.csv'
LABEL_NAMES   = ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']
LABEL_MAP     = {name: i for i, name in enumerate(LABEL_NAMES)}
BANGLABERT_ID = 'csebuetnlp/banglabert'
RANDOM_SEED   = 42
EPOCHS        = 10
BATCH_SIZE    = 16   # smaller than ViT -- BERT is memory-hungry
LR            = 2e-5  # standard BERT fine-tune LR
MAX_LEN       = 128
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -- Reproducibility ----------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[+] Random seed fixed: {seed} (deterministic mode ON)")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -- Dataset (Text-only, no image) --------------------------------------------
class TextOnlyDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=MAX_LEN):
        self.df        = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len   = max_len

        # Detect caption column
        for col in ['bangla_caption', 'caption_bn', 'caption', 'text']:
            if col in self.df.columns:
                self.text_col = col
                break
        else:
            raise ValueError(
                f"No caption column found. Available: {list(self.df.columns)}"
            )
        print(f"[+] Using text column: '{self.text_col}'")
        self.df[self.text_col] = self.df[self.text_col].fillna('').astype(str)
        self.df['label_id'] = self.df['macro_label'].map(LABEL_MAP)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = str(row[self.text_col])
        encoding = self.tokenizer(
            text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(int(row['label_id']), dtype=torch.long)
        }


# -- Stratified Split (identical to all other scripts) ------------------------
def get_split_indices(csv_path, random_state=42):
    df      = pd.read_csv(csv_path)
    labels  = df['macro_label'].values
    all_idx = list(range(len(df)))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.2, stratify=labels, random_state=random_state
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=random_state
    )
    return train_idx, val_idx, test_idx


# -- Model (BanglaBERT + Classification Head) ---------------------------------
class BanglaBERTOnlyClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        print(f"[+] Loading BanglaBERT ({BANGLABERT_ID})...")
        self.bert = AutoModel.from_pretrained(BANGLABERT_ID)

        # Freeze all layers first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze last 2 transformer blocks + pooler (name-based)
        for name, param in self.bert.named_parameters():
            if ("encoder.layer.10" in name or
                "encoder.layer.11" in name or
                "pooler"           in name):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.bert.parameters())
        print(f"[+] Partial fine-tune -- Trainable: {trainable:,} / {total:,} params "
              f"({100*trainable/total:.2f}%)")

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]   # [B, 768]
        return self.classifier(cls_emb)


# -- Metrics helper -----------------------------------------------------------
def print_cls_report(title, y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro',  zero_division=0) * 100
    macro_p  = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    macro_r  = recall_score(y_true, y_pred, average='macro',  zero_division=0) * 100
    per_f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_p    = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_r    = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm       = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    w = 62
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")
    print(f"  Macro F1        : {macro_f1:.2f}%")
    print(f"  Macro Precision : {macro_p:.2f}%")
    print(f"  Macro Recall    : {macro_r:.2f}%\n")
    print("  Per-class breakdown:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"    [{name:<22}] P={per_p[i]:.4f}  R={per_r[i]:.4f}  F1={per_f1[i]:.4f}")
    print()
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    header = f"{'':>16}" + "".join(f"  {n[:8]:>8}" for n in LABEL_NAMES)
    print(f"  {header}")
    for i, rl in enumerate(["Severe_Damag", "Humanitarian", "Affected_Peo"]):
        row_str = "  ".join(f"{cm[i][j]:>8}" for j in range(3))
        print(f"  {rl:<14}  {row_str}")
    print(f"{'='*w}")
    return macro_f1


# -- Training loop ------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        labels    = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attn_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        labels    = batch['label']
        preds     = model(input_ids, attn_mask).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


# -- Main ---------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=EPOCHS)
    parser.add_argument('--batch_size', type=int,   default=BATCH_SIZE)
    parser.add_argument('--lr',         type=float, default=LR)
    parser.add_argument('--seed',       type=int,   default=RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"\n{'='*62}")
    print(f"  DRISHTI-XAI -- Task 0.6: BanglaBERT-Only Baseline (No Image)")
    print(f"  Device: {DEVICE}  |  Seed: {args.seed}")
    print(f"{'='*62}\n")

    # -- Tokenizer + Dataset --------------------------------------------------
    print(f"[+] Loading tokenizer ({BANGLABERT_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(BANGLABERT_ID)

    dataset = TextOnlyDataset(CSV_PATH, tokenizer, max_len=MAX_LEN)
    train_idx, val_idx, test_idx = get_split_indices(CSV_PATH, args.seed)
    print(f"[+] Split: Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # -- Model ----------------------------------------------------------------
    model     = BanglaBERTOnlyClassifier(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -- Training -------------------------------------------------------------
    print(f"\n[+] Training for {args.epochs} epochs...\n")
    best_val_f1, best_state, best_epoch = 0.0, None, 0

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        val_labels, val_preds = evaluate(model, val_loader)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0) * 100

        print(f"  Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Loss={loss:.4f} | Train Acc={train_acc*100:.2f}% | "
              f"Val Macro-F1={val_f1:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n[+] Best checkpoint: Epoch {best_epoch} (Val Macro-F1={best_val_f1:.2f}%)")

    # -- Test Evaluation ------------------------------------------------------
    model.load_state_dict(best_state)
    y_true, y_pred = evaluate(model, test_loader)
    f1 = print_cls_report(
        "Task 0.6 -- BanglaBERT-Only Baseline (TEST SET)",
        y_true, y_pred
    )

    # -- Comparison summary ---------------------------------------------------
    w = 62
    print(f"\n{'='*w}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*w}")
    print(f"  {'Method':<36} {'Macro F1':>8}")
    print(f"  {'-'*36} {'-'*8}")
    print(f"  {'Majority Classifier':<36} {'24.89%':>8}")
    print(f"  {'Stratified Random':<36} {'39.95%':>8}")
    print(f"  {'BanglaBERT-Only (this script)':<36} {f1:>7.2f}%  <- Task 0.6")
    print(f"  {'ViT-Only':<36} {'78.64%':>8}  <- Task 0.5")
    print(f"  {'DisasterNet-v2 (ViT + BanglaBERT)':<36} {'77.95%':>8}  <- Task 0.3")
    print(f"{'='*w}")
    print(f"\n[!] Image encoder contribution (ViT-Only vs BERT-Only):  {78.64 - f1:+.2f}pp")
    print(f"[!] Multimodal gain over text-only (v2 vs BERT-Only):    {77.95 - f1:+.2f}pp")
    print()


if __name__ == '__main__':
    main()
